#!/usr/bin/env python
"""Fail-fast architecture gate: memorize 32 fixed examples, no held-out claims."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.config import Config
from src.models.factory import MODEL_NAMES, build_model, parameter_count
from src.data.factory import build_data
from src.training.distributed import setup, synchronize
from src.training.metrics import loss_and_counts


def run(models=MODEL_NAMES, max_steps=1500, device="auto", threshold=0.05, output="results/overfit.json", task="associative_recall"):
    cfg = Config(task=task, embedding_dim=32, hidden_dim=128, latent_dim=128, field_width=128,
                 transformer_ff=128, batch_size=32, seq_len=32, vocab_symbols=8, recall_pairs=4,
                 device=device, amp=False, seed=123, train_steps=max_steps,
                 learning_rate=0.003, weight_decay=0.0, warmup_steps=0)
    _, _, selected_device = setup(cfg)
    data = build_data(cfg)
    x, y = data.batch(32, 32, 7001)
    x, y = x.to(selected_device), y.to(selected_device)
    results = []
    for name in models:
        torch.manual_seed(cfg.seed)
        model = build_model(replace(cfg, model=name), data.vocab_size).to(selected_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        synchronize(selected_device)
        start = time.perf_counter()
        passed = False
        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss_sum, correct, count = loss_and_counts(logits, y)
            loss = loss_sum/count
            if not torch.isfinite(loss):
                break
            if loss.item() < threshold:
                passed = True
                break
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                break
            optimizer.step()
            if (step+1)%100 == 0:
                print(f"{name}: step={step+1} loss={loss.item():.5f}", flush=True)
        synchronize(selected_device)
        record = dict(model=name, task=task, seed=cfg.seed, steps=step+1, loss=loss.item(),
                      accuracy=(correct/count).item(), passed=passed, batch_generator_seed=7001, schedule="constant", seconds=time.perf_counter()-start,
                      parameters=parameter_count(model), threshold=threshold, examples=32, length=32,
                      config=replace(cfg, model=name).to_dict())
        results.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "config"}), flush=True)
        del model, optimizer
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    if not all(r["passed"] for r in results):
        raise RuntimeError("Overfit gate failed; inspect results before launching experiments")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task", default="associative_recall")
    parser.add_argument("--output", default="results/overfit.json")
    args = parser.parse_args()
    run(args.models, args.max_steps, args.device, args.threshold, args.output, args.task)
