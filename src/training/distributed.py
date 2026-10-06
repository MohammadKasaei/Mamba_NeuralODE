import os
import random
import numpy as np
import torch
from torch import distributed as dist


def setup(cfg):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = torch.cuda.is_available() and cfg.device != "cpu"
    device = torch.device(f"cuda:{local}" if cuda else "cpu") if cfg.device == "auto" or world > 1 else torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    # Same initial weights on all ranks; data RNGs are independent by explicit batch seed.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return rank, world, device


def reduce_sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
