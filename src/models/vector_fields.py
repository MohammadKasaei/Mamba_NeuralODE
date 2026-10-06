import torch
from torch import nn
from torch.nn import functional as F


class VectorField(nn.Module):
    """W2 SiLU(W1 (Wh LN(h) + Wx x)), optional gating and damping."""
    def __init__(self, state_dim, input_dim, width, damping=False, gated=False):
        super().__init__()
        self.norm = nn.LayerNorm(state_dim)
        self.wh = nn.Linear(state_dim, width, bias=False)
        self.wx = nn.Linear(input_dim, width, bias=False) if input_dim else None
        self.w1 = nn.Linear(width, width)
        self.w2 = nn.Linear(width, state_dim)
        self.gate = nn.Linear(width, state_dim) if gated else None
        self.decay = nn.Parameter(torch.full((state_dim,), -2.0)) if damping else None

    def forward(self, h, x=None):
        q = self.wh(self.norm(h))
        if self.wx is not None:
            q = q + self.wx(x)
        out = self.w2(F.silu(self.w1(q)))
        if self.gate is not None:
            out = out * torch.sigmoid(self.gate(q))
        if self.decay is not None:
            out = out - F.softplus(self.decay) * h
        return out


class DiagonalField(nn.Module):
    """Diagonal A(x), and either b(x) or a low-rank input-dependent B(x)x."""
    def __init__(self, state_dim, input_dim, a_conditioned=True, b_conditioned=True,
                 stable=True, drive="bias"):
        super().__init__()
        self.a_bias = nn.Parameter(torch.full((state_dim,), -1.0))
        self.a_map = nn.Linear(input_dim, state_dim, bias=False) if a_conditioned else None
        self.b_map = nn.Linear(input_dim, state_dim, bias=False) if b_conditioned or drive == "matrix" else None
        self.b_bias = nn.Parameter(torch.zeros(state_dim))
        self.b_gate = nn.Linear(input_dim, state_dim) if drive == "matrix" and b_conditioned else None
        self.stable, self.drive = stable, drive

    def coefficients(self, x):
        a = self.a_bias.expand(x.shape[0], -1)
        if self.a_map is not None:
            a = a + self.a_map(x)
        a = -F.softplus(a) if self.stable else a
        b = self.b_bias.expand(x.shape[0], -1)
        if self.b_map is not None:
            bx = self.b_map(x)
            if self.b_gate is not None:
                bx = torch.sigmoid(self.b_gate(x)) * bx
            b = b + bx
        return a, b

    def forward(self, h, x):
        a, b = self.coefficients(x)
        return a * h + b
