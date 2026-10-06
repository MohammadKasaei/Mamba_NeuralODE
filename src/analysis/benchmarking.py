import time
import torch
from torch import nn
from src.models.gru import GRULM
from src.models.lstm import LSTMLM
from src.models.transformer import TransformerLM
from src.training.trainer import autocast
from src.training.distributed import synchronize
from src.training.metrics import loss_and_counts
from src.models.factory import parameter_count


@torch.no_grad()
def estimate_flops(model, x):
    """Dominant forward multiply-add FLOPs, not a hardware instruction counter.

    Linear modules counted from actual shapes; fused RNN and MHA explicitly added.
    Ignores elementwise ops, embeddings, normalization and softmax; no backward estimate.
    """
    count = [0]
    hooks = []
    def linear_hook(module, args, out):
        count[0] += 2*args[0].numel()*module.out_features
    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    try:
        model(x)
    finally:
        for hook in hooks:
            hook.remove()
    batch, length = x.shape
    if isinstance(model, (GRULM, LSTMLM)):
        gates = 3 if isinstance(model, GRULM) else 4
        count[0] += batch*length*2*gates*model.cfg.hidden_dim*(model.cfg.hidden_dim+model.cfg.embedding_dim)
    if isinstance(model, TransformerLM):
        d = model.cfg.embedding_dim
        count[0] += model.cfg.transformer_layers*(8*batch*length*d*d + 4*batch*length*length*d)
    return count[0]/x.numel()


def benchmark(model, x, y, cfg, device, warmup=3, repeats=10, initial_scale=65536.0):
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats >= 1 and warmup >= 0 required")
    results = {"parameters": parameter_count(model), "nfe_per_token": model.nfe_per_token,
               "batch_size": x.size(0), "seq_len": x.size(1), "amp_fp16": cfg.amp and device.type == "cuda",
               "forward_flops_per_token_estimate": estimate_flops(model, x),
               "flops_scope": "dominant dense multiply-adds; excludes elementwise, backward and optimizer",
               "timing_scope": "single device, resident batch; training includes optimizer, excludes data generation"}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda", init_scale=initial_scale)
    skipped = [0]
    for training in (False, True):
        model.train(training)
        def iteration():
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), autocast(cfg, device):
                logits = model(x)
                if training:
                    loss, _, count = loss_and_counts(logits, y)
                    loss = loss/count
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss during benchmark")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped[0] += int(scaler.get_scale() < before)
        for _ in range(warmup):
            iteration()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        skips_before = skipped[0]
        start = time.perf_counter()
        for _ in range(repeats):
            iteration()
        synchronize(device)
        elapsed = time.perf_counter()-start
        name = "train" if training else "inference"
        results[f"{name}_tokens_per_sec"] = x.numel()*repeats/elapsed
        results[f"{name}_ms_per_token"] = elapsed*1000/(x.numel()*repeats)
        if training:
            results["train_skipped_updates"] = skipped[0]-skips_before
            results["amp_scale"] = scaler.get_scale()
        results[f"{name}_peak_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    model.eval()
    return results
