"""On-demand, deterministic batches with next-token labels and task-only scoring.

Vocabulary: PAD=0, MARK=1, QUERY=2, COPY=3, keys/data start at 4.
Each sequence is generated at L+1 then shifted, so the final answer is never
visible at the position being scored. All copying examples have copy_items targets.
"""
import torch

TASKS = ("selective_copying", "associative_recall", "induction")

class SyntheticData:
    def __init__(self, cfg):
        if cfg.task not in TASKS:
            raise ValueError(f"Unknown synthetic task {cfg.task}")
        self.cfg = cfg
        self.vocab_size = 4 + 2*cfg.vocab_symbols
        self.metadata = {"task": cfg.task, "vocab_size": self.vocab_size,
                         "scoring": "answer tokens only", "data_seed": cfg.seed}

    def batch(self, batch_size, length, seed, split="train"):
        cfg = self.cfg
        gen = torch.Generator().manual_seed(int(seed))
        full = torch.zeros(batch_size, length+1, dtype=torch.long)
        mask = torch.zeros_like(full, dtype=torch.bool)
        for row in range(batch_size):
            if cfg.task == "selective_copying":
                m = cfg.copy_items
                # Each selected symbol has a preceding MARK; other symbols are distractors.
                prefix = length-m
                if prefix < 2*m+1:
                    raise ValueError("Selective copy length must be >= 3*copy_items+1")
                full[row, :prefix] = torch.randint(4, 4+cfg.vocab_symbols, (prefix,), generator=gen)
                slots = torch.randperm((prefix-1)//2, generator=gen)[:m].sort().values * 2
                selected = torch.randint(4, 4+cfg.vocab_symbols, (m,), generator=gen)
                full[row, slots] = 1
                full[row, slots+1] = selected
                full[row, prefix] = 3
                full[row, prefix+1:] = selected
                mask[row, prefix+1:] = True
            elif cfg.task == "associative_recall":
                p = cfg.recall_pairs
                if p > cfg.vocab_symbols or length < 2*p+3:
                    raise ValueError("Recall needs enough unique keys and length >= 2*pairs+3")
                keys = torch.randperm(cfg.vocab_symbols, generator=gen)[:p] + 4
                values = torch.randint(4+cfg.vocab_symbols, self.vocab_size, (p,), generator=gen)
                slots = torch.randperm((length-2)//2, generator=gen)[:p].sort().values*2
                full[row, slots], full[row, slots+1] = keys, values
                query = torch.randint(p, (), generator=gen).item()
                full[row, length-2] = keys[query]
                full[row, length-1] = 2
                full[row, length] = values[query]
                mask[row, length] = True
            else:
                # Random shared-vocabulary stream, one unambiguous previous q,v bigram.
                # q appears exactly once in context, then once as the final query.
                if length < 4:
                    raise ValueError("Induction needs length >= 4")
                symbols = cfg.vocab_symbols
                if symbols < 2:
                    raise ValueError("Induction needs >= 2 symbols")
                q = torch.randint(symbols, (), generator=gen).item()
                distractors = torch.randint(symbols-1, (length+1,), generator=gen)
                distractors += (distractors >= q).long()
                full[row] = distractors+4
                pos = torch.randint(length-2, (), generator=gen).item()
                full[row, pos] = q+4
                answer = full[row, pos+1].clone()
                full[row, length-1], full[row, length] = q+4, answer
                mask[row, length] = True
        x, y = full[:, :-1], full[:, 1:].clone()
        y[~mask[:, 1:]] = -100
        return x, y
