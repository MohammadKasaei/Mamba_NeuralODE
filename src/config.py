"""Strict, serializable experiment configuration; CLI overrides use YAML values."""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import yaml

@dataclass
class Config:
    model: str = "conditioned"
    task: str = "associative_recall"
    seed: int = 0
    embedding_dim: int = 128
    hidden_dim: int = 256
    field_width: int = 256
    latent_dim: int = 256
    integrator: str = "euler"
    steps_per_token: int = 1
    damping: bool = False
    gated: bool = False
    a_conditioned: bool = True
    b_conditioned: bool = True
    stable_a: bool = True
    structured_drive: str = "bias"  # bias=b(x), matrix=B(x)x (low-rank)
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff: int = 256
    dropout: float = 0.0
    batch_size: int = 16  # per rank, per accumulation microbatch
    accumulation_steps: int = 1
    seq_len: int = 64
    eval_lengths: tuple = (128, 256, 512, 1024, 2048)
    train_steps: int = 1000
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.001
    early_stopping_min_steps: int = 1000
    eval_every: int = 100
    eval_batches: int = 8
    eval_batch_size: int = 0  # zero uses the training microbatch
    eval_token_budget: int = 8192  # maximum tokens per long-context test microbatch
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0
    amp: bool = True
    device: str = "auto"
    deterministic: bool = True
    vocab_symbols: int = 16
    copy_items: int = 4
    recall_pairs: int = 8
    data_path: str = "data_cache/tiny_shakespeare.txt"
    language_train_fraction: float = 0.9
    language_val_fraction: float = 0.1
    output_dir: str = "results/run"
    resume: str = ""
    match_parameters: int = 0
    match_tolerance: float = 0.10
    diagnostics_examples: int = 1
    diagnostics_tokens: int = 64

    def validate(self):
        for key in ("embedding_dim", "hidden_dim", "field_width", "latent_dim", "batch_size",
                    "accumulation_steps", "seq_len", "train_steps", "eval_every", "eval_batches",
                    "steps_per_token", "vocab_symbols", "copy_items", "recall_pairs"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.integrator not in ("euler", "heun", "rk4"):
            raise ValueError("Unknown integrator")
        if self.model == "residual" and self.integrator != "euler":
            raise ValueError("The residual control always uses Euler; set integrator=euler")
        if self.structured_drive not in ("bias", "matrix"):
            raise ValueError("structured_drive must be bias or matrix")
        if self.embedding_dim % self.transformer_heads and self.model == "transformer":
            raise ValueError("embedding_dim must divide transformer_heads")
        if self.eval_lengths and min(self.eval_lengths) < 1:
            raise ValueError("eval_lengths must be positive")
        if self.early_stopping_patience < 0 or self.early_stopping_min_delta < 0 or self.early_stopping_min_steps < 0:
            raise ValueError("Early stopping settings must be nonnegative")
        if self.eval_batch_size < 0 or self.eval_token_budget < 1:
            raise ValueError("Evaluation batch size must be nonnegative and token budget positive")
        if not (0 < self.language_train_fraction < 1 and 0 < self.language_val_fraction < 1
                and self.language_train_fraction + self.language_val_fraction <= 1 + 1e-12):
            raise ValueError("Language split fractions must be positive and sum to at most one")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def evaluation_batch_size(self):
        return self.eval_batch_size or self.batch_size


def load_config(path=None, overrides=()):
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    data = data or {}
    for entry in overrides:
        key, value = entry.split("=", 1)
        data[key] = yaml.safe_load(value)
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    return Config(**data).validate()
