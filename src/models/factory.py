from dataclasses import replace
from .gru import GRULM
from .lstm import LSTMLM
from .conditioned_node import ConditionedNODE, ResidualRNN
from .augmented_node import AugmentedNODE
from .projected_node import ProjectedNODE
from .structured_node import StructuredNODE
from .transformer import TransformerLM

MODEL_NAMES = ("gru", "lstm", "residual", "autonomous", "conditioned", "augmented", "projected_conditioned", "structured", "transformer")


def build_model(cfg, vocab_size):
    constructors = dict(gru=GRULM, lstm=LSTMLM, residual=ResidualRNN, conditioned=ConditionedNODE,
                        augmented=AugmentedNODE, structured=StructuredNODE, transformer=TransformerLM)
    if cfg.model in ("autonomous", "projected_conditioned"):
        return ProjectedNODE(vocab_size, cfg, cfg.model == "projected_conditioned")
    if cfg.model not in constructors:
        raise ValueError(f"Unknown model {cfg.model}; choices: {MODEL_NAMES}")
    return constructors[cfg.model](vocab_size, cfg)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_capacity(cfg, vocab_size):
    """Search actual parameter counts, preserving embedding dim and latent/state widths for NODEs.

    GRU/LSTM/structured: hidden size; generic NODEs: field width; Transformer: FF width.
    Strict matching fails rather than silently claiming fairness.
    """
    if not cfg.match_parameters:
        return cfg
    key = ("hidden_dim" if cfg.model in ("gru", "lstm", "structured") else
           "transformer_ff" if cfg.model == "transformer" else "field_width")
    lo, hi = 4, 4096
    candidates = []
    while lo <= hi:
        mid = (lo + hi)//2
        candidate = replace(cfg, **{key: mid})
        count = parameter_count(build_model(candidate, vocab_size))
        candidates.append((abs(count-cfg.match_parameters), count, candidate))
        if count < cfg.match_parameters:
            lo = mid+1
        else:
            hi = mid-1
    _, count, best = min(candidates, key=lambda item: item[0])
    if abs(count/cfg.match_parameters - 1) > cfg.match_tolerance:
        raise ValueError(f"Cannot match {cfg.model} to {cfg.match_parameters}: nearest {count}. "
                         "Change target or embedding/latent size; do not silently relax fairness.")
    return best
