from .synthetic import SyntheticData
from .tiny_shakespeare import ShakespeareData

def build_data(cfg):
    return ShakespeareData(cfg) if cfg.task == "shakespeare" else SyntheticData(cfg)
