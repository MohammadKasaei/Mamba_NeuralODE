#!/usr/bin/env python
"""Evaluate a selected checkpoint on independent task data and memory controls."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.data.factory import build_data
from src.data.interventions import ChangedRecallQuery, RemovedMemoryCue
from src.models.factory import build_model
from src.training.distributed import setup, cleanup
from src.training.trainer import evaluate, load_checkpoint
from src.analysis.dynamics import diagnose

TEST_SEED_OFFSET = 1000000000000


def run(checkpoint):
    ckpt = load_checkpoint(checkpoint)
    cfg = Config(**ckpt["config"])
    rank, world, device = setup(cfg)
    if world != 1:
        raise ValueError("Independent-test protocol requires one GPU per experiment")
    try:
        data = build_data(cfg)
        if data.metadata != ckpt["data"]:
            raise ValueError("Evaluation data differs from training")
        split = "test" if cfg.task == "shakespeare" else "val"
        model = build_model(cfg, data.vocab_size).to(device)
        model.load_state_dict(ckpt["model"])
        metrics = []
        for length in sorted(set([cfg.seq_len, *cfg.eval_lengths])):
            # Hold the number of examples fixed when long contexts need smaller batches.
            batch = min(cfg.evaluation_batch_size, max(1, cfg.eval_token_budget//length))
            examples = cfg.evaluation_batch_size*cfg.eval_batches
            if examples % batch:
                raise ValueError("Evaluation microbatch must divide the requested example count")
            evaluation = replace(cfg, batch_size=batch, eval_batch_size=0, eval_batches=examples//batch)
            try:
                point = evaluate(model, data, evaluation, device, length=length,
                                 seed_offset=TEST_SEED_OFFSET, split=split)
                point["finite"] = True
            except FloatingPointError as error:
                point = {"seq_len": length, "finite": False, "error": str(error)}
            metrics.append(point)
            print(json.dumps(point), flush=True)
        control = None
        if cfg.task != "shakespeare":
            changed = ChangedRecallQuery(data, cfg.vocab_symbols) if cfg.task == "associative_recall" else RemovedMemoryCue(data, cfg.task)
            try:
                control = evaluate(model, changed, cfg, device, length=cfg.seq_len,
                                   seed_offset=TEST_SEED_OFFSET, split=split)
                control["finite"] = True
            except FloatingPointError as error:
                control = {"finite": False, "error": str(error)}
            control["description"] = {"associative_recall": "Change query to another present key; retain original labels",
                                      "selective_copying": "Remove MARK tokens; retain original labels and teacher-forced outputs",
                                      "induction": "Remove the earlier query occurrence; retain original labels"}[cfg.task]
        output = Path(checkpoint).parent
        payload = {"checkpoint": str(checkpoint), "step": ckpt["step"], "seed_offset": TEST_SEED_OFFSET,
                   "split": split, "metrics": metrics, "memory_control": control}
        temporary = output/"test_results.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(output/"test_results.json")
        x, _ = data.batch(cfg.diagnostics_examples, cfg.seq_len, 17001, split)
        diagnose(model.float(), x[:, :cfg.diagnostics_tokens].to(device), output/"dynamics")
        return payload
    finally:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    run(parser.parse_args().checkpoint)
