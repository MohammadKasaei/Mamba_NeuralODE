"""DDP/FP16 trainer with stateless data generation, resumable checkpoints, and timed metrics."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import random
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from zipfile import ZipFile, ZIP_DEFLATED
from src.models.factory import build_model, match_capacity, parameter_count
from src.data.factory import build_data
from .distributed import setup, cleanup, reduce_sum, synchronize
from .metrics import loss_and_counts, quality


def autocast(cfg, device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda")


def batch_seed(cfg, step, micro=0, rank=0, validation=False, length=0):
    # Disjoint train/validation streams; validation examples identical across architectures.
    return cfg.seed*1000000007 + (100000000000 if validation else 0) + step*1000003 + micro*1009 + rank*9176 + length*37


@torch.no_grad()
def evaluate(model, data, cfg, device, rank=0, world=1, length=None, seed_offset=0, split="val"):
    if cfg.eval_batch_size:
        cfg = replace(cfg, batch_size=cfg.eval_batch_size, eval_batch_size=0)
    was_training = model.training
    model.eval()
    length = length or cfg.seq_len
    totals = torch.zeros(7, device=device, dtype=torch.float64)
    synchronize(device)
    start = time.perf_counter()
    for i in range(cfg.eval_batches):
        x, y = data.batch(cfg.batch_size, length, batch_seed(cfg, i, rank=rank, validation=True, length=length) + seed_offset, split)
        x, y = x.to(device), y.to(device)
        with autocast(cfg, device):
            logits, _, stats = model(x, return_stats=True)
            loss, correct, count = loss_and_counts(logits, y)
        exact = ((logits.argmax(-1) == y) | (y == -100)).all(dim=1).sum()
        totals += torch.stack((loss.double(), correct.double(), count.double(),
                               stats["hidden_norm"].double(), stats["update_norm"].double(),
                               exact.double(), count.new_tensor(x.size(0)).double()))
    synchronize(device)
    elapsed = time.perf_counter()-start
    reduce_sum(totals)
    elapsed_t = torch.tensor(elapsed, device=device)
    if dist.is_initialized():
        dist.all_reduce(elapsed_t, op=dist.ReduceOp.MAX)
    elapsed = elapsed_t.item()
    loss, correct, count, hn, un, exact, examples = totals.tolist()
    if not math.isfinite(loss) or not math.isfinite(hn) or not math.isfinite(un):
        raise FloatingPointError(f"Nonfinite evaluation at sequence length {length}")
    result = quality(loss/count, correct/count)
    result.update(seq_len=length, hidden_norm=hn/(cfg.eval_batches*world),
                  update_norm=un/(cfg.eval_batches*world), inference_tokens_per_sec=cfg.eval_batches*cfg.batch_size*length*world/elapsed,
                  inference_ms_per_token=elapsed*1000/(cfg.eval_batches*cfg.batch_size*length*world),
                  supervised_tokens=int(count), inference_seconds=elapsed,
                  examples=int(examples), exact_sequence_accuracy=exact/examples)
    model.train(was_training)
    return result


def runtime_metadata(device):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "python": platform.python_version(),
            "torch": torch.__version__, "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(path, model, optimizer, scaler, cfg, data, step, best, world, training_seconds,
                    stopping_state=None):
    # Every rank contributes its RNG state (important for Transformer dropout).
    states = [None]*world
    local = rng_state()
    if dist.is_initialized():
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if not dist.is_initialized() or dist.get_rank() == 0:
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "config": cfg.to_dict(), "step": step,
                   "best_loss": best, "vocab_size": data.vocab_size, "data": data.metadata,
                   "rng_states": states, "world_size": world, "training_seconds": training_seconds,
                   "stopping_state": stopping_state or {}}
        temporary = Path(str(path)+".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Only load trusted checkpoints: optimizer/RNG snapshots contain Python objects.
    return torch.load(path, map_location=device, weights_only=False)


def train(cfg):
    rank, world, device = setup(cfg)
    try:
        data = build_data(cfg)
        cfg = match_capacity(cfg, data.vocab_size)
        # Capacity search creates temporary models; reset RNG so embeddings share initialization.
        torch.manual_seed(cfg.seed)
        model = build_model(cfg, data.vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
        start_step, best, training_seconds = 0, float("inf"), 0.0
        stopping_state = {"anchor_loss": None, "stale_checks": 0, "skipped_updates": 0, "nan_inf_events": 0}
        if cfg.resume:
            ckpt = load_checkpoint(cfg.resume, device)
            if ckpt["world_size"] != world:
                raise ValueError("Exact resume requires the same world size")
            from src.config import Config
            previous = Config(**ckpt["config"]).to_dict()
            allowed = {"resume", "output_dir", "device", "diagnostics_examples", "diagnostics_tokens"}
            changed = [k for k, v in cfg.to_dict().items() if k not in allowed and v != previous[k]]
            if changed or ckpt["data"] != data.metadata:
                raise ValueError(f"Resume requires identical experiment/data; changed fields: {changed}")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            start_step, best = ckpt["step"], ckpt["best_loss"]
            training_seconds = ckpt.get("training_seconds", 0.0)
            stopping_state.update(ckpt.get("stopping_state", {}))
        unused_embedding = cfg.model == "structured" and not cfg.a_conditioned and not cfg.b_conditioned and cfg.structured_drive == "bias"
        wrapper = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=unused_embedding) if world > 1 else model
        if cfg.resume:
            restore_rng(ckpt["rng_states"][rank])
        output = Path(cfg.output_dir)
        if not cfg.resume and (output/"final.pt").exists():
            raise FileExistsError(f"Run already exists: {output}; select a new output_dir or resume")
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            (output/"config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
            meta = runtime_metadata(device)
            sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for folder in ("src", "scripts", "tests") for p in sorted(Path(folder).rglob("*.py"))}
            (output/"source_hashes.json").write_text(json.dumps(sources, indent=2))
            with ZipFile(output/"source_snapshot.zip", "w", compression=ZIP_DEFLATED) as archive:
                for source in sources:
                    archive.write(source, arcname=source)
                for source in ("requirements.txt", "pyproject.toml", "README.md"):
                    if Path(source).exists():
                        archive.write(source, arcname=source)
            meta.update(seed=cfg.seed, parameters=parameter_count(model), world_size=world,
                        effective_batch=cfg.batch_size*world*cfg.accumulation_steps,
                        nfe_per_token=model.nfe_per_token, data=data.metadata)
            (output/"metadata.json").write_text(json.dumps(meta, indent=2))
        if dist.is_initialized():
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics_file = output/"metrics.csv"
        # Discard rows newer than the atomic checkpoint after an interrupted run.
        if cfg.resume and rank == 0 and metrics_file.exists():
            with metrics_file.open(newline="") as file:
                reader = csv.DictReader(file)
                columns = reader.fieldnames
                rows = [row for row in reader if int(row["step"]) <= start_step]
            with metrics_file.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            if rows and not ckpt.get("stopping_state"):
                stopping_state.update(skipped_updates=int(rows[-1]["skipped_updates"]),
                                      nan_inf_events=int(rows[-1]["nan_inf_events"]))
        nan_events, skipped_updates = stopping_state["nan_inf_events"], stopping_state["skipped_updates"]
        last_eval = None
        early_stopped = False
        step = start_step-1
        finished_resume = bool(cfg.resume and (start_step >= cfg.train_steps or (
            cfg.early_stopping_patience and stopping_state["stale_checks"] >= cfg.early_stopping_patience)))
        if finished_resume:
            last_eval = evaluate(model, data, cfg, device, rank, world)
            early_stopped = bool(cfg.early_stopping_patience and stopping_state["stale_checks"] >= cfg.early_stopping_patience)
        for step in (() if finished_resume else range(start_step, cfg.train_steps)):
            wrapper.train()
            optimizer.zero_grad(set_to_none=True)
            if step < cfg.warmup_steps:
                scale = (step+1)/max(cfg.warmup_steps, 1)
            else:
                progress = (step-cfg.warmup_steps)/max(cfg.train_steps-cfg.warmup_steps-1, 1)
                scale = 0.5*(1+math.cos(math.pi*progress))
            lr = cfg.learning_rate*scale
            for group in optimizer.param_groups:
                group["lr"] = lr
            sums = torch.zeros(5, device=device, dtype=torch.float64)
            synchronize(device)
            started = time.perf_counter()
            # The same effective batch is used when microbatch/accumulation partitions change.
            all_x, all_y = data.batch(cfg.batch_size*cfg.accumulation_steps, cfg.seq_len, batch_seed(cfg, step, rank=rank))
            for micro in range(cfg.accumulation_steps):
                lo, hi = micro*cfg.batch_size, (micro+1)*cfg.batch_size
                x, y = all_x[lo:hi], all_y[lo:hi]
                x, y = x.to(device), y.to(device)
                sync_context = wrapper.no_sync() if world > 1 and micro < cfg.accumulation_steps-1 else nullcontext()
                with sync_context:
                    with autocast(cfg, device):
                        logits, _, stats = wrapper(x, return_stats=True)
                        loss, correct, count = loss_and_counts(logits, y)
                        normalized = loss/count/cfg.accumulation_steps
                    finite = torch.isfinite(normalized).to(torch.int32)
                    if dist.is_initialized():
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        nan_events += 1
                        raise FloatingPointError(f"Nonfinite loss at step {step}, microbatch {micro}")
                    scaler.scale(normalized).backward()
                sums += torch.stack((loss.detach().double(), correct.double(), count.double(),
                                     stats["hidden_norm"].double(), stats["update_norm"].double()))
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            finite_grad = torch.isfinite(grad).to(torch.int32)
            if dist.is_initialized():
                dist.all_reduce(finite_grad, op=dist.ReduceOp.MIN)
            if not finite_grad.item():
                nan_events += 1
                if not scaler.is_enabled():
                    raise FloatingPointError(f"Nonfinite gradients at step {step}")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_updates += int(scaler.get_scale() < old_scale)
            synchronize(device)
            elapsed = torch.tensor(time.perf_counter()-started, device=device)
            if dist.is_initialized():
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            elapsed = elapsed.item()
            training_seconds += elapsed
            reduce_sum(sums)
            sl, sc, sn, sh, su = sums.tolist()
            peak = torch.tensor(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            row = {"step": step+1, "train_loss": sl/sn, "train_accuracy": sc/sn,
                   "learning_rate": lr, "gradient_norm": grad.item(),
                   "hidden_norm": sh/(world*cfg.accumulation_steps), "update_norm": su/(world*cfg.accumulation_steps),
                   "nan_inf_events": nan_events, "skipped_updates": skipped_updates,
                   "amp_scale": scaler.get_scale(), "training_seconds": training_seconds,
                   "train_tokens_per_sec": cfg.batch_size*cfg.seq_len*cfg.accumulation_steps*world/elapsed,
                   "train_ms_per_token": elapsed*1000/(cfg.batch_size*cfg.seq_len*cfg.accumulation_steps*world),
                   "peak_memory_bytes": int(peak.item()), "val_loss": "", "val_bpc": "", "val_accuracy": ""}
            if (step+1) % cfg.eval_every == 0 or step+1 == cfg.train_steps:
                # Evaluate underlying module, without DDP forward collectives.
                last_eval = evaluate(model, data, cfg, device, rank, world)
                row.update(val_loss=last_eval["loss"], val_bpc=last_eval["bpc"], val_accuracy=last_eval["accuracy"])
                stopping_state.update(skipped_updates=skipped_updates, nan_inf_events=nan_events)
                if step+1 >= cfg.early_stopping_min_steps:
                    anchor = stopping_state["anchor_loss"]
                    if anchor is None or last_eval["loss"] < anchor-cfg.early_stopping_min_delta:
                        stopping_state.update(anchor_loss=last_eval["loss"], stale_checks=0)
                    else:
                        stopping_state["stale_checks"] += 1
                    early_stopped = bool(cfg.early_stopping_patience and
                                         stopping_state["stale_checks"] >= cfg.early_stopping_patience)
                if last_eval["loss"] < best:
                    best = last_eval["loss"]
                    save_checkpoint(output/"best.pt", model, optimizer, scaler, cfg, data, step+1, best, world, training_seconds, stopping_state)
                save_checkpoint(output/"final.pt", model, optimizer, scaler, cfg, data, step+1, best, world, training_seconds, stopping_state)
                if rank == 0:
                    print(f"{cfg.model} step={step+1} train={sl/sn:.4f} val={best:.4f} tokens/s={row['train_tokens_per_sec']:.0f}", flush=True)
            if rank == 0:
                exists = metrics_file.exists()
                with metrics_file.open("a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
            if early_stopped:
                if rank == 0:
                    print(f"Early stopping at {step+1}: {stopping_state['stale_checks']} checks without improvement", flush=True)
                break
        if rank == 0 and last_eval is not None:
            (output/"validation.json").write_text(json.dumps(last_eval, indent=2))
            (output/"training_summary.json").write_text(json.dumps({
                "status": "early_stopped" if early_stopped else "budget_complete",
                "attempted_updates": step+1, "successful_updates": step+1-skipped_updates,
                "skipped_updates": skipped_updates, "nan_inf_events": nan_events,
                "best_validation_loss": best, "training_seconds": training_seconds,
                "stopping_state": stopping_state}, indent=2))
            (output/"failure.json").unlink(missing_ok=True)
        return cfg
    except Exception as error:
        if rank == 0:
            failure_dir = Path(cfg.output_dir)
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir/"failure.json").write_text(json.dumps({"error_type": type(error).__name__, "message": str(error), "nan_inf_events": locals().get("nan_events", 0), "step": locals().get("step")}, indent=2))
        raise
    finally:
        cleanup()
