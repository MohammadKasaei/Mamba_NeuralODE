"""Plan reproducible one-factor ablations, then optionally execute them sequentially.

Four GPUs distribute each independent experiment's batches, not model parallelism.
No combinatorial mixing of unrelated ablations. Every planned run is saved as YAML.
"""
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.config import load_config
from src.models.factory import MODEL_NAMES, build_model, parameter_count
from src.models.integrators import NFE_PER_STEP
from src.data.factory import build_data

ODE_ABLATIONS = ("autonomous", "conditioned", "projected_conditioned", "structured")


def plans(base, suite, tasks, seeds, lengths):
    for task, seed, length in itertools.product(tasks, seeds, lengths):
        cfg = replace(base, task=task, seed=seed, seq_len=length)
        # Capacity target is the default GRU count; main comparisons matched +/-10%.
        vocab = build_data(cfg).vocab_size
        target = parameter_count(build_model(replace(cfg, model="gru", match_parameters=0), vocab))
        if suite in ("main", "full", "pilot"):
            for name in MODEL_NAMES:
                yield "main", replace(cfg, model=name, integrator="euler" if name == "residual" else cfg.integrator, match_parameters=target)
        # Full suite repeats solver/state ablations at the representative training length;
        # main models still train at all requested lengths.
        if suite == "full" and length != base.seq_len:
            continue
        if suite in ("integration", "full"):
            for name, method, k in itertools.product(ODE_ABLATIONS, ("euler", "heun", "rk4"), (1, 2, 4, 8)):
                yield "integration", replace(cfg, model=name, integrator=method, steps_per_token=k, match_parameters=target)
            for k in (1, 2, 4, 8):
                yield "integration", replace(cfg, model="residual", integrator="euler", steps_per_token=k, match_parameters=target)
        if suite in ("state", "full"):
            # Fixed states are intentionally unmatched: capacity is the manipulated variable.
            for name, size in itertools.product(tuple(m for m in MODEL_NAMES if m != "transformer"), (128, 256, 512)):
                yield "state", replace(cfg, model=name, integrator="euler" if name == "residual" else cfg.integrator, hidden_dim=size, latent_dim=size, field_width=size, match_parameters=0)
        if suite in ("conditioning", "full"):
            for ac, bc, stable, drive in itertools.product((False, True), (False, True), (False, True), ("bias", "matrix")):
                yield "conditioning", replace(cfg, model="structured", a_conditioned=ac, b_conditioned=bc,
                                                stable_a=stable, structured_drive=drive, match_parameters=0)
        if suite in ("stability", "full"):
            for damping, gated in itertools.product((False, True), repeat=2):
                yield "stability", replace(cfg, model="conditioned", damping=damping, gated=gated, match_parameters=target)


def main(default_config, language=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--suite", choices=("main", "integration", "state", "conditioning", "stability", "full", "pilot"), default="main")
    parser.add_argument("--tasks", nargs="+", default=["shakespeare"] if language else ["selective_copying", "associative_recall", "induction"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--lengths", nargs="+", type=int, default=[256] if language else [64, 128, 256])
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--output", default="results/language_sweep" if language else "results/synthetic_sweep")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Execute/plan only the first N unique configurations")
    args = parser.parse_args()
    base = load_config(args.config, args.set)
    if args.nproc < 1 or args.nproc > 4:
        parser.error("nproc must be 1..4")
    if args.suite == "pilot":
        # Deliberate small engineering pilot; never substitutes for 3-seed experiments.
        base = replace(base, embedding_dim=16, hidden_dim=32, latent_dim=32, field_width=32,
                       transformer_ff=64, batch_size=16, train_steps=120, warmup_steps=5,
                       eval_every=40, eval_batches=4, vocab_symbols=8, recall_pairs=4,
                       eval_lengths=(32, 64, 128), learning_rate=0.003)
        args.tasks = ["associative_recall"]
        args.seeds, args.lengths = [0], [32]
    if args.suite == "full" and base.seq_len not in args.lengths:
        parser.error("Full suite requires base seq_len in --lengths; include it or override seq_len")
    root = Path(args.output)
    configs = root/"configs"
    configs.mkdir(parents=True, exist_ok=True)
    unique = {}
    for group, cfg in plans(base, args.suite, args.tasks, args.seeds, args.lengths):
        identity = cfg.to_dict()
        identity.pop("output_dir")
        identity.pop("resume")
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        if digest in unique:
            unique[digest]["groups"].append(group)
            continue
        run = root/f"{cfg.task}_{cfg.model}_s{cfg.seed}_L{cfg.seq_len}_{digest}"
        # Bound retained solver graphs while preserving effective batch and examples.
        nfe = cfg.steps_per_token * (1 if cfg.model == "residual" else NFE_PER_STEP[cfg.integrator]) if cfg.model in (*ODE_ABLATIONS, "residual", "augmented") else 1
        activation_budget = base.batch_size * base.seq_len * 128
        effective_batch = cfg.batch_size * cfg.accumulation_steps
        candidates = [b for b in range(1, cfg.batch_size+1) if effective_batch % b == 0 and b*cfg.seq_len*nfe <= activation_budget]
        microbatch = max(candidates) if candidates else 1
        cfg = replace(cfg, batch_size=microbatch, accumulation_steps=effective_batch//microbatch, output_dir=str(run), resume="")
        cfg.validate()
        path = configs/f"{digest}.yaml"
        path.write_text(yaml.safe_dump(cfg.to_dict()))
        unique[digest] = dict(groups=[group], config=str(path), run=str(run))
        if args.limit and len(unique) >= args.limit:
            break
    manifest = list(unique.values())
    (root/"plan.json").write_text(json.dumps(manifest, indent=2))
    print(f"Planned {len(manifest)} experiments in {root}; execute={args.execute}", flush=True)
    if not args.execute:
        return
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OMP_NUM_THREADS="2")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], env=environment, check=True)
    gate = root/"overfit.json"
    gate_results = json.loads(gate.read_text()) if gate.exists() else []
    if {r["model"] for r in gate_results} != set(MODEL_NAMES) or not all(r["passed"] for r in gate_results):
        subprocess.run([sys.executable, "scripts/overfit.py", "--output", str(gate)], env=environment, check=True)
    for entry in manifest:
        run = Path(entry["run"])
        config = yaml.safe_load(Path(entry["config"]).read_text())
        if (run/"failure.json").exists() and json.loads((run/"failure.json").read_text())["error_type"] == "FloatingPointError":
            print(f"Retaining recorded numerical failure: {run}", flush=True)
            continue
        finished = False
        if (run/"final.pt").exists():
            from src.training.trainer import load_checkpoint
            finished = load_checkpoint(run/"final.pt")["step"] >= config["train_steps"]
        if not finished:
            train_args = ["scripts/train.py", "--config", entry["config"]]
            if (run/"final.pt").exists():
                train_args += ["--set", f"resume={run/'final.pt'}"]
            launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.nproc}"] if args.nproc > 1 else [sys.executable]
            outcome = subprocess.run(launcher+train_args, env=environment)
            if outcome.returncode:
                failure = run/"failure.json"
                if failure.exists() and json.loads(failure.read_text())["error_type"] == "FloatingPointError":
                    print(f"Numerical failure recorded; continuing other configurations: {run}", flush=True)
                    continue
                raise subprocess.CalledProcessError(outcome.returncode, launcher+train_args)
        for script, artifact in (("eval.py", "extrapolation.json"), ("benchmark.py", "benchmark.json")):
            if not (run/artifact).exists():
                subprocess.run([sys.executable, f"scripts/{script}", str(run/"best.pt")], env=environment, check=True)
    subprocess.run([sys.executable, "scripts/analyze.py", "--root", str(root)], env=environment, check=True)
