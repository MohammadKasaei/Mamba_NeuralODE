#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config, load_config
from src.models.factory import build_model
from src.data.factory import build_data
from src.training.distributed import setup, cleanup
from src.training.trainer import load_checkpoint
from src.analysis.benchmarking import benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--config", help="Benchmark an untrained architecture for resource planning")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--output", help="JSON output path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.config):
        parser.error("Supply exactly one checkpoint or --config")
    ckpt = load_checkpoint(args.checkpoint) if args.checkpoint else None
    cfg = Config(**ckpt["config"]) if ckpt else load_config(args.config, args.set)
    if args.device:
        cfg.device = args.device
    rank, world, device = setup(cfg)
    try:
        if world != 1:
            raise ValueError("Benchmark uses one GPU for comparable model costs; use train metrics for DDP throughput")
        data = build_data(cfg)
        model = build_model(cfg, data.vocab_size).to(device)
        if ckpt:
            model.load_state_dict(ckpt["model"])
        model.eval()
        x, y = data.batch(args.batch_size or cfg.batch_size, args.length or cfg.seq_len, 19001, "val")
        result = benchmark(model, x.to(device), y.to(device), cfg, device, args.warmup, args.repeats, initial_scale=ckpt["scaler"].get("scale", 65536.0) if ckpt else 65536.0)
        result.update(checkpoint=args.checkpoint, trained=ckpt is not None, device=str(device),
                      model=cfg.model, integrator=cfg.integrator, K=cfg.steps_per_token,
                      config=cfg.to_dict())
        output = Path(args.output) if args.output else (Path(args.checkpoint).parent/"benchmark.json" if ckpt else Path(cfg.output_dir)/"resource_probe.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        cleanup()
