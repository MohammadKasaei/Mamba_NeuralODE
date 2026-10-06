"""Evaluate a single trained vector field at different numerical resolutions.

This is separate from retraining with each solver: it probes whether the learned
field defines useful dynamics independently of its training discretization.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.data.factory import build_data
from src.models.factory import build_model
from src.training.trainer import load_checkpoint, evaluate
from src.training.distributed import setup, cleanup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--methods", nargs="+", default=["euler", "heun", "rk4"])
    parser.add_argument("--batches", type=int)
    args = parser.parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    cfg = Config(**ckpt["config"])
    if cfg.model in ("gru", "lstm", "transformer"):
        parser.error("Solver refinement requires an ODE/residual model")
    if args.batches:
        cfg.eval_batches = args.batches
    rank, world, device = setup(cfg)
    try:
        data = build_data(cfg)
        rows = []
        for method in args.methods:
            for k in args.steps:
                refined = replace(cfg, model="conditioned" if cfg.model == "residual" else cfg.model,
                                  integrator=method, steps_per_token=k).validate()
                model = build_model(refined, data.vocab_size).to(device)
                model.load_state_dict(ckpt["model"])
                row = evaluate(model, data, refined, device, rank, world)
                row.update(integrator=method, K=k, nfe_per_token=model.nfe_per_token)
                rows.append(row)
        if rank == 0:
            (Path(args.checkpoint).parent/"solver_refinement.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(rows, indent=2))
    finally:
        cleanup()
