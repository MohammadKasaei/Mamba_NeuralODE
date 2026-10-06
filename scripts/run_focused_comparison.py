#!/usr/bin/env python
"""Run the explicitly requested three-seed architecture and solver comparisons.

Independent single-GPU training processes share a queue across GPUs. This avoids
DDP communication for compact cells and keeps effective batch/data identical.
Training phases precede independent-test evaluation and isolated benchmarking.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.config import load_config
from src.data.factory import build_data
from src.models.factory import build_model, match_capacity, parameter_count
from src.training.trainer import load_checkpoint

ARCHITECTURES = ("conditioned", "autonomous", "structured", "gru")
TEST_SEED_OFFSET = 1000000000000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/focused_recall.yaml")
    parser.add_argument("--output", default="results/focused_recall")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        parser.error("Use unique GPU indices")
    cfg = load_config(args.config, args.set)
    if cfg.task != "associative_recall":
        parser.error("This focused protocol is defined for associative recall")
    root = Path(args.output)
    (root/"configs").mkdir(parents=True, exist_ok=True)
    (root/"logs").mkdir(exist_ok=True)
    vocab = build_data(cfg).vocab_size
    target = parameter_count(build_model(replace(cfg, model="gru", match_parameters=0), vocab))
    entries = []
    for phase in ("architecture", "solver"):
        combinations = [(m, "euler", 1) for m in ARCHITECTURES] if phase == "architecture" else [("conditioned", "euler", 4), ("conditioned", "heun", 4)]
        for model, method, k in combinations:
            for seed in args.seeds:
                name = f"{model}_{method}_k{k}_s{seed}"
                run = root/name
                resolved = match_capacity(replace(cfg, model=model, integrator=method, steps_per_token=k,
                                                    seed=seed, match_parameters=target, output_dir=str(run), resume=""), vocab)
                resolved.validate()
                path = root/"configs"/f"{name}.yaml"
                if path.exists() and yaml.safe_load(path.read_text()) != yaml.safe_load(yaml.safe_dump(resolved.to_dict())):
                    raise ValueError(f"Existing experiment differs: {path}; use a new output directory")
                path.write_text(yaml.safe_dump(resolved.to_dict()))
                entries.append(dict(name=name, phase=phase, config=str(path), run=str(run),
                                    parameters=parameter_count(build_model(resolved, vocab))))
    protocol = dict(config=cfg.to_dict(), seeds=args.seeds, architecture_models=list(ARCHITECTURES),
                    solver_settings=[["euler", 1], ["euler", 4], ["heun", 4]], parameters_target=target,
                    world_size=1, effective_batch=cfg.batch_size*cfg.accumulation_steps,
                    test_seed_offset=TEST_SEED_OFFSET, examples_per_test_length=cfg.batch_size*cfg.eval_batches,
                    hardware_policy="One independent experiment per GPU; same global batch for every model",
                    checkpoint_selection="minimum fixed-stream validation CE; report independent test accuracy",
                    entries=entries)
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2))
    print(f"Planned {len(entries)} runs; parameters target={target}; execute={args.execute}", flush=True)
    if not args.execute:
        return
    environment = dict(os.environ, OMP_NUM_THREADS="2", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    with (root/"logs/preflight.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, "scripts/overfit.py", "--output", str(root/"overfit.json")],
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    statuses = {}
    lock = threading.Lock()
    def update(name, status, **details):
        with lock:
            statuses[name] = dict(status=status, **details)
            temporary = root/"progress.json.tmp"
            temporary.write_text(json.dumps(statuses, indent=2))
            temporary.replace(root/"progress.json")
        print(f"{name}: {status}", flush=True)
    def worker(gpu, jobs):
        env = dict(environment, CUDA_VISIBLE_DEVICES=str(gpu))
        while True:
            try:
                entry = jobs.get_nowait()
            except queue.Empty:
                return
            name, run = entry["name"], Path(entry["run"])
            config = yaml.safe_load(Path(entry["config"]).read_text())
            if (run/"final.pt").exists() and load_checkpoint(run/"final.pt")["step"] >= config["train_steps"]:
                update(name, "trained", gpu=gpu, reused=True)
                continue
            command = [sys.executable, "scripts/train.py", "--config", entry["config"]]
            if (run/"final.pt").exists():
                command += ["--set", f"resume={run/'final.pt'}"]
            update(name, "training", gpu=gpu, started=time.time())
            with (root/"logs"/f"{name}.train.log").open("a") as log:
                outcome = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if outcome.returncode:
                update(name, "failed", gpu=gpu, returncode=outcome.returncode)
                if not (run/"failure.json").exists() or json.loads((run/"failure.json").read_text())["error_type"] != "FloatingPointError":
                    raise subprocess.CalledProcessError(outcome.returncode, command)
            else:
                update(name, "trained", gpu=gpu)
    for phase in ("architecture", "solver"):
        jobs = queue.Queue()
        for entry in entries:
            if entry["phase"] == phase:
                jobs.put(entry)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [executor.submit(worker, gpu, jobs) for gpu in args.gpus]
            for future in futures:
                future.result()
    # Test evaluation can run in parallel on separate GPUs; all training is done.
    jobs = queue.Queue()
    for entry in entries:
        if statuses.get(entry["name"], {}).get("status") != "failed":
            jobs.put(entry)
    def test_worker(gpu):
        env = dict(environment, CUDA_VISIBLE_DEVICES=str(gpu))
        while True:
            try:
                entry = jobs.get_nowait()
            except queue.Empty:
                return
            run = Path(entry["run"])
            if not (run/"extrapolation.json").exists():
                update(entry["name"], "testing", gpu=gpu)
                with (root/"logs"/f"{entry['name']}.eval.log").open("w") as log:
                    subprocess.run([sys.executable, "scripts/eval.py", str(run/"best.pt"),
                                    "--seed-offset", str(TEST_SEED_OFFSET)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            update(entry["name"], "tested", gpu=gpu)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(test_worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    # Measure every architecture/solver on the same GPU, with no concurrent experiments.
    env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.gpus[0]))
    for entry in entries:
        if statuses.get(entry["name"], {}).get("status") == "failed":
            continue
        run = Path(entry["run"])
        if not (run/"benchmark.json").exists():
            with (root/"logs"/f"{entry['name']}.benchmark.log").open("w") as log:
                subprocess.run([sys.executable, "scripts/benchmark.py", str(run/"best.pt"), "--warmup", "3", "--repeats", "10"],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        update(entry["name"], "complete", benchmark_gpu=args.gpus[0])
    subprocess.run([sys.executable, "scripts/analyze.py", "--root", str(root)], env=env, check=True)
    from src.analysis.focused import report
    report(root)

if __name__ == "__main__":
    main()
