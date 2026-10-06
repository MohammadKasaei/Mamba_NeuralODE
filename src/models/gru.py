from torch import nn
from .base import RecurrentLM

class GRULM(RecurrentLM):
    def __init__(self, vocab_size, cfg):
        super().__init__(vocab_size, cfg)
        self.cell = nn.GRUCell(cfg.embedding_dim, cfg.hidden_dim)

    def transition(self, x, h):
        return self.cell(x, h)
