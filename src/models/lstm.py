import torch
from torch import nn
from .base import RecurrentLM

class LSTMLM(RecurrentLM):
    def __init__(self, vocab_size, cfg):
        super().__init__(vocab_size, cfg)
        self.cell = nn.LSTMCell(cfg.embedding_dim, cfg.hidden_dim)

    def initial_state(self, batch, device, dtype):
        h = super().initial_state(batch, device, dtype)
        return h, torch.zeros_like(h)

    def transition(self, x, state):
        return self.cell(x, state)
