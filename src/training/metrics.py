import math
import torch
from torch.nn import functional as F


def loss_and_counts(logits, targets):
    loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-100, reduction="sum")
    valid = targets != -100
    count = valid.sum()
    if count.item() == 0:
        raise ValueError("Batch has no supervised tokens")
    correct = ((logits.argmax(-1) == targets) & valid).sum()
    return loss_sum, correct, count


def quality(loss, accuracy):
    return {"loss": loss, "accuracy": accuracy, "bpc": loss/math.log(2),
            "perplexity": math.exp(min(loss, 80))}
