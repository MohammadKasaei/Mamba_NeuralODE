import torch
from torch import nn
from .base import ODELM
from .integrators import integrate
from .vector_fields import VectorField

class ProjectedNODE(ODELM):
    def __init__(self, vocab_size, cfg, conditioned=False):
        super().__init__(vocab_size, cfg)
        self.conditioned = conditioned
        self.project_in = nn.Linear(cfg.hidden_dim + cfg.embedding_dim, cfg.latent_dim)
        self.project_out = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.field = VectorField(cfg.latent_dim, cfg.embedding_dim if conditioned else 0,
                                 cfg.field_width, cfg.damping, cfg.gated)

    def latent_solve(self, s, x, trajectory=False):
        return integrate(lambda y: self.field(y, x if self.conditioned else None), s,
                         self.cfg.steps_per_token, self.cfg.integrator, trajectory)

    def transition(self, x, h):
        # Keep solver state and carried memory FP32, as in the directly conditioned solve.
        s = self.project_in(torch.cat((h, x), -1)).float()
        return self.project_out(self.latent_solve(s, x)).float()
