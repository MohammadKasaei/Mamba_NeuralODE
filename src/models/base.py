import torch
from torch import nn
from .integrators import NFE_PER_STEP


class RecurrentLM(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim)
        self.readout = nn.Linear(cfg.hidden_dim, vocab_size)

    def initial_state(self, batch, device, dtype):
        return torch.zeros(batch, self.cfg.hidden_dim, device=device, dtype=dtype)

    def memory(self, state):
        return state[0] if isinstance(state, tuple) else state

    def forward(self, tokens, state=None, return_stats=False):
        x = self.embedding(tokens)
        if state is None:
            state = self.initial_state(tokens.size(0), x.device, x.dtype)
        outputs, norms, updates = [], [], []
        for xt in x.unbind(1):
            before = self.memory(state)
            state = self.transition(xt, state)
            h = self.memory(state)
            outputs.append(self.readout(h))
            if return_stats:
                norms.append(h.detach().float().norm(dim=-1).mean())
                updates.append((h.detach().float()-before.detach().float()).norm(dim=-1).mean())
        logits = torch.stack(outputs, 1)
        if return_stats:
            return logits, state, {"hidden_norm": torch.stack(norms).mean(),
                                   "update_norm": torch.stack(updates).mean()}
        return logits

    @property
    def nfe_per_token(self):
        return 0


class ODELM(RecurrentLM):
    @property
    def nfe_per_token(self):
        return self.cfg.steps_per_token * NFE_PER_STEP[self.cfg.integrator]
