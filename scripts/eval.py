#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.models.factory import build_model
from src.data.factory import build_data
from src.training.distributed import setup, cleanup
from src.training.trainer import load_checkpoint, evaluate
from src.analysis.dynamics import diagnose


def run(checkpoint, lengths=None, batches=None, device=None, diagnostics=True, seed_offset=0, split="val"):
    ckpt = load_checkpoint(checkpoint)
    cfg = Config(**ckpt["config"])
    if batches is not None:
        cfg.eval_batches = batches
    if device:
        cfg.device = device
    cfg.validate()
    rank, world, selected = setup(cfg)
    try:
        data = build_data(cfg)
        if data.metadata != ckpt["data"]:
            raise ValueError("Evaluation data/vocabulary differs from checkpoint")
        model = build_model(cfg, data.vocab_size).to(selected)
        model.load_state_dict(ckpt["model"])
        results = []
        for length in (lengths or sorted(set([cfg.seq_len]+list(cfg.eval_lengths)))):
            try:
                item = evaluate(model, data, cfg, selected, rank, world, length, seed_offset=seed_offset, split=split)
                item["finite"] = True
            except FloatingPointError as error:
                item = {"seq_len": length, "finite": False, "loss": None, "accuracy": None, "bpc": None, "error": str(error)}
            results.append(item)
        intervention = None
        if seed_offset and cfg.task == "associative_recall" and cfg.recall_pairs > 1:
            from src.data.interventions import ChangedRecallQuery
            intervention = evaluate(model, ChangedRecallQuery(data, cfg.vocab_symbols), cfg, selected, rank, world,
                                    cfg.seq_len, seed_offset=seed_offset, split=split)
            intervention["description"] = "query replaced by another present key; original labels retained"
        output = Path(checkpoint).parent
        if rank == 0:
            (output/"extrapolation.json").write_text(json.dumps({"checkpoint": str(checkpoint), "step": ckpt["step"], "seed_offset": seed_offset, "split": split, "query_ablation": intervention, "metrics": results}, indent=2))
            if diagnostics:
                x, _ = data.batch(cfg.diagnostics_examples, cfg.seq_len, 17001, split)
                diagnose(model.float(), x[:, :cfg.diagnostics_tokens].to(selected), output/"dynamics")
            print(json.dumps(results, indent=2))
        return results
    finally:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--batches", type=int)
    parser.add_argument("--device")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=0, help="Independent synthetic holdout sampling; LM still samples validation text")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    run(args.checkpoint, args.lengths, args.batches, args.device, not args.no_diagnostics, args.seed_offset, args.split)
