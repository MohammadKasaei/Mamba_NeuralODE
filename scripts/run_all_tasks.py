#!/usr/bin/env python
"""Resumable nine-architecture, three-seed study of all four implemented tasks.

Training and held-out evaluation run on separate GPUs in a shared job queue.
Isolated compute benchmarks run only after the full training queue is finished.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
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
from src.models.factory import MODEL_NAMES, build_model, match_capacity, parameter_count
from src.analysis.all_tasks import report, TEST_SEED_OFFSET

TASKS = ("selective_copying", "associative_recall", "induction", "shakespeare")


def plan(args):
    root = Path(args.output).resolve()
    (root/"configs").mkdir(parents=True, exist_ok=True)
    (root/"logs").mkdir(exist_ok=True)
    entries = []
    bases = {}
    for task in args.tasks:
        base = load_config(args.language_config if task == "shakespeare" else args.synthetic_config, args.set)
        base = replace(base, task=task)
        vocab = build_data(base).vocab_size
        target = parameter_count(build_model(replace(base, model="gru", match_parameters=0), vocab))
        bases[task] = dict(config=base.to_dict(), parameters_target=target)
        for model in args.models:
            for seed in args.seeds:
                name = f"{task}_{model}_s{seed}"
                run = root/task/f"{model}_s{seed}"
                cfg = match_capacity(replace(base, model=model, seed=seed, match_parameters=target,
                                             output_dir=str(run), resume=""), vocab).validate()
                path = root/"configs"/f"{name}.yaml"
                if path.exists() and yaml.safe_load(path.read_text()) != cfg.to_dict():
                    raise ValueError(f"Existing configuration differs: {path}; use another output directory")
                path.write_text(yaml.safe_dump(cfg.to_dict()))
                entries.append(dict(name=name, task=task, model=model, seed=seed, config=str(path), run=str(run),
                                    parameters=parameter_count(build_model(cfg, vocab)), config_values=cfg.to_dict()))
    # Round-robin tasks so results from all tasks start arriving early.
    entries.sort(key=lambda e: (args.models.index(e["model"]), args.seeds.index(e["seed"]), args.tasks.index(e["task"])))
    protocol = dict(tasks=args.tasks, models=args.models, seeds=args.seeds, task_protocols=bases, entries=entries,
                    test_seed_offset=TEST_SEED_OFFSET, world_size=1,
                    hardware_policy="One independent single-GPU job per device; equal global batch within each task",
                    checkpoint_selection="Minimum fixed-stream validation CE; early stopping uses validation only",
                    scope="Main architectures, Euler K=1; solver/state/conditioning ablations are separate studies")
    path = root/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing protocol differs; use a new output directory")
    path.write_text(json.dumps(protocol, indent=2))
    return root, entries


def execute(args, root, entries):
    environment = dict(os.environ, OMP_NUM_THREADS="2", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    lock_file = (root/"runner.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A runner already owns this output directory") from error
    (root/"runner.json").write_text(json.dumps(dict(pid=os.getpid(), gpus=args.gpus,
                                                    started=datetime.now(timezone.utc).isoformat()), indent=2))
    def command(arguments, log, gpu=None):
        env = dict(environment)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with Path(log).open("a") as file:
            outcome = subprocess.run([sys.executable, *arguments], env=env, stdout=file, stderr=subprocess.STDOUT)
        if outcome.returncode:
            raise subprocess.CalledProcessError(outcome.returncode, arguments)
    if not args.skip_preflight:
        print("Running tests and all synthetic overfit gates", flush=True)
        command(["-m", "pytest", "-q"], root/"logs/preflight.log", args.gpus[0])
        for task in (task for task in args.tasks if task != "shakespeare"):
            output = root/f"overfit_{task}.json"
            if output.exists() and all(r["passed"] for r in json.loads(output.read_text())):
                continue
            command(["scripts/overfit.py", "--task", task, "--models", *args.models, "--output", str(output)],
                    root/"logs/preflight.log", args.gpus[0])
    if not args.skip_probes:
        probes = root/"resource_probes"
        probes.mkdir(exist_ok=True)
        representatives = [e for e in entries if e["seed"] == args.seeds[0] and e["task"] in ("associative_recall", "shakespeare")]
        print(f"Measuring {len(representatives)} isolated resource probes", flush=True)
        for entry in representatives:
            path = probes/f"{entry['task']}_{entry['model']}.json"
            if path.exists():
                continue
            command(["scripts/benchmark.py", "--config", entry["config"], "--warmup", "3", "--repeats", "10", "--state-statistics", "--output", str(path)],
                    root/"logs/resource_probes.log", args.gpus[0])
        rates = {}
        for entry in representatives:
            rate = json.loads((probes/f"{entry['task']}_{entry['model']}.json").read_text())["train_tokens_per_sec"]
            rates[(entry["task"], entry["model"])] = rate
        seconds = 0
        for entry in entries:
            cfg = entry["config_values"]
            rate = rates.get((entry["task"], entry["model"]), rates.get(("associative_recall", entry["model"])))
            if rate:
                seconds += cfg["train_steps"]*cfg["batch_size"]*cfg["accumulation_steps"]*cfg["seq_len"]/rate
        estimate = dict(maximum_budget_training_gpu_hours=seconds/3600,
                        ideal_queue_hours_at_maximum_budget=seconds/3600/len(args.gpus),
                        note="Compute-only estimate at 20,000 updates; excludes data generation, validation, testing, IO and CPU contention. Early stopping can shorten runs. Untrained probes are not quality observations.")
        (root/"runtime_estimate.json").write_text(json.dumps(estimate, indent=2))
        print(json.dumps(estimate), flush=True)
    if args.probes_only:
        print("Preflight/probes complete; training not launched", flush=True)
        return
    statuses = json.loads((root/"progress.json").read_text()) if (root/"progress.json").exists() else {}
    lock = threading.Lock()
    report_lock = threading.Lock()
    def update(entry, status, **details):
        with lock:
            statuses[entry["name"]] = dict(status=status, updated=time.time(), **details)
            temporary = root/"progress.json.tmp"
            temporary.write_text(json.dumps(statuses, indent=2))
            temporary.replace(root/"progress.json")
        print(f"{entry['name']}: {status}", flush=True)
    jobs = queue.Queue()
    for entry in entries:
        jobs.put(entry)
    def worker(gpu):
        while True:
            try:
                entry = jobs.get_nowait()
            except queue.Empty:
                return
            run = Path(entry["run"])
            try:
                done = (run/"training_summary.json").exists() and json.loads((run/"training_summary.json").read_text())["status"] in ("early_stopped", "budget_complete")
                if not done:
                    arguments = ["scripts/train.py", "--config", entry["config"]]
                    if (run/"final.pt").exists():
                        arguments += ["--set", f"resume={run/'final.pt'}"]
                    update(entry, "training", gpu=gpu)
                    command(arguments, root/"logs"/f"{entry['name']}.train.log", gpu)
                update(entry, "trained", gpu=gpu)
                if not (run/"test_results.json").exists() or not (run/"dynamics/diagnostics.json").exists():
                    update(entry, "testing", gpu=gpu)
                    command(["scripts/eval_all_tasks.py", str(run/"best.pt")], root/"logs"/f"{entry['name']}.test.log", gpu)
                update(entry, "tested", gpu=gpu)
                # Publish interim reports after each block of 12 completed jobs.
                with lock:
                    completed = sum(s["status"] in ("tested", "complete") for s in statuses.values())
                if completed % 12 == 0:
                    with report_lock:
                        report(root, plots=False)
            except subprocess.CalledProcessError as error:
                failure = json.loads((run/"failure.json").read_text()) if (run/"failure.json").exists() else {}
                update(entry, "failed", gpu=gpu, returncode=error.returncode, error=failure)
                if failure.get("error_type") != "FloatingPointError":
                    raise
    # Produce initial reports, with explicit pending observations and task baselines.
    report(root, plots=False)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    report(root)
    print("Training/testing finished; beginning isolated benchmarks", flush=True)
    for entry in entries:
        if statuses[entry["name"]]["status"] == "failed":
            continue
        run = Path(entry["run"])
        if not (run/"benchmark.json").exists():
            update(entry, "benchmarking", gpu=args.gpus[0])
            command(["scripts/benchmark.py", str(run/"best.pt"), "--warmup", "3", "--repeats", "10"],
                    root/"logs"/f"{entry['name']}.benchmark.log", args.gpus[0])
        update(entry, "complete", benchmark_gpu=args.gpus[0])
    rows = report(root)
    failed = [r["run"] for r in rows if r["status"] == "failed"]
    (root/"completion.json").write_text(json.dumps(dict(status="complete_with_failures" if failed else "complete",
                                                       planned=len(entries), tested=sum(bool(r.get("tested")) for r in rows),
                                                       failed=failed, finished=datetime.now(timezone.utc).isoformat()), indent=2))
    print(f"Study finished: {root/'summary/report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-config", default="configs/all_tasks_synthetic.yaml")
    parser.add_argument("--language-config", default="configs/all_tasks_language.yaml")
    parser.add_argument("--output", default="results/all_tasks_large_batch")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probes-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true", help="Only reuse already verified code/gates")
    parser.add_argument("--skip-probes", action="store_true")
    args = parser.parse_args()
    for name in ("tasks", "models", "seeds", "gpus"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"{name} must contain unique values")
    root, entries = plan(args)
    print(f"Planned {len(entries)} runs at {root}; execute={args.execute}", flush=True)
    if args.execute:
        execute(args, root, entries)


if __name__ == "__main__":
    main()
