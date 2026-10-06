import math
import torch
from torch import nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim)
        layer = nn.TransformerEncoderLayer(cfg.embedding_dim, cfg.transformer_heads,
                cfg.transformer_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.transformer_layers, enable_nested_tensor=False)
        self.readout = nn.Linear(cfg.embedding_dim, vocab_size)

    def forward(self, tokens, state=None, return_stats=False):
        if state is not None:
            raise ValueError("Transformer has no recurrent carry; supply full context")
        x = self.embedding(tokens)
        length, dim = x.shape[1:]
        position = torch.arange(length, device=x.device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) * (-math.log(10000)/dim))
        pe = torch.zeros(length, dim, device=x.device)
        pe[:, 0::2] = torch.sin(position*freq)
        pe[:, 1::2] = torch.cos(position*freq[:dim//2])
        mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(x + pe.to(x.dtype), mask=mask, is_causal=True)
        logits = self.readout(h)
        if return_stats:
            return logits, None, {"hidden_norm": h.detach().float().norm(dim=-1).mean(),
                                  "update_norm": (h-x).detach().float().norm(dim=-1).mean()}
        return logits

    @property
    def nfe_per_token(self):
        return 0
