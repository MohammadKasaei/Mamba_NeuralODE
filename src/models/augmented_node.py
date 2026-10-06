import torch
from .conditioned_node import ConditionedNODE
from .integrators import integrate

class AugmentedNODE(ConditionedNODE):
    def augmented_field(self, z):
        h, x = z.split((self.cfg.hidden_dim, self.cfg.embedding_dim), dim=-1)
        return torch.cat((self.field(h, x), torch.zeros_like(x)), dim=-1)

    def solve_augmented(self, h, x, trajectory=False):
        return integrate(self.augmented_field, torch.cat((h, x), -1), self.cfg.steps_per_token,
                         self.cfg.integrator, trajectory)

    def solve(self, h, x, trajectory=False):
        result = self.solve_augmented(h, x, trajectory)
        if trajectory:
            z, path = result
            return z[..., :self.cfg.hidden_dim], [p[..., :self.cfg.hidden_dim] for p in path]
        return result[..., :self.cfg.hidden_dim]
