#!/usr/bin/env python
"""Measure batch scaling with the state-statistics kernels used during training."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.config import load_config
from src.data.factory import build_data
from src.models.factory import build_model
from src.training.distributed import setup, cleanup
from src.analysis.benchmarking import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    _, world, device = setup(cfg)
    if world != 1:
        raise ValueError("Batch calibration requires one GPU")
    data = build_data(cfg)
    observations = []
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for batch in args.batches:
            model = None
            x = y = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            try:
                torch.manual_seed(cfg.seed)
                model = build_model(cfg, data.vocab_size).to(device)
                timings = []
                for _ in range(3):
                    start = time.perf_counter()
                    x, y = data.batch(batch, cfg.seq_len, 19001, "val")
                    timings.append(time.perf_counter()-start)
                x, y = x.to(device), y.to(device)
                result = benchmark(model, x, y, replace(cfg, batch_size=batch), device,
                                   warmup=3, repeats=args.repeats, initial_scale=64.0, return_stats=True)
                generation_seconds = sorted(timings)[1]
                train_seconds = batch*cfg.seq_len/result["train_tokens_per_sec"]
                result.update(status="ok", task=cfg.task, model=cfg.model, data_generation_seconds=generation_seconds,
                              estimated_tokens_per_sec_with_generation=batch*cfg.seq_len/(train_seconds+generation_seconds),
                              trained=False, initial_amp_scale=64.0)
            except torch.OutOfMemoryError as error:
                result = dict(status="out_of_memory", batch_size=batch, model=cfg.model, task=cfg.task, error=str(error))
            observations.append(result)
            path.write_text(json.dumps(observations, indent=2))
            print(json.dumps({k: v for k, v in result.items() if k not in ("error", "flops_scope", "timing_scope")}), flush=True)
            del model, x, y
    finally:
        cleanup()


if __name__ == "__main__":
    main()
