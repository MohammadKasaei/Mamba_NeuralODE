from pathlib import Path
import hashlib
import urllib.request
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
    return path


class ShakespeareData:
    def __init__(self, cfg):
        path = Path(cfg.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Download first: python scripts/download_data.py --path {path}")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        vocab = sorted(set(text))
        mapping = {c: i for i, c in enumerate(vocab)}
        tokens = torch.tensor([mapping[c] for c in text], dtype=torch.long)
        cutoff = int(cfg.language_train_fraction*len(tokens))
        end = int((cfg.language_train_fraction + cfg.language_val_fraction)*len(tokens))
        end = min(end, len(tokens))
        self.splits = {"train": tokens[:cutoff], "val": tokens[cutoff:end]}
        if end < len(tokens):
            self.splits["test"] = tokens[end:]
        split = "contiguous 90% train / 10% validation" if (cfg.language_train_fraction, cfg.language_val_fraction) == (0.9, 0.1) else (
            f"contiguous {100*cfg.language_train_fraction:g}% train / "
            f"{100*cfg.language_val_fraction:g}% validation / "
            f"{100*(1-cfg.language_train_fraction-cfg.language_val_fraction):g}% test")
        self.vocab_size = len(vocab)
        self.metadata = {"sha256": hashlib.sha256(raw).hexdigest(), "vocab": vocab,
                         "split": split, "path": str(path)}

    def batch(self, batch_size, length, seed, split="train"):
        if split not in self.splits:
            raise ValueError(f"No {split!r} split; configure a reserved test fraction for test evaluation")
        tokens = self.splits[split]
        if len(tokens) <= length:
            raise ValueError("Requested context exceeds split length")
        gen = torch.Generator().manual_seed(int(seed))
        starts = torch.randint(len(tokens)-length, (batch_size,), generator=gen)
        blocks = torch.stack([tokens[i:i+length+1] for i in starts])
        return blocks[:, :-1], blocks[:, 1:]
