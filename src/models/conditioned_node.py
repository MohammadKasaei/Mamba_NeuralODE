from .base import ODELM
from .integrators import integrate
from .vector_fields import VectorField

class ConditionedNODE(ODELM):
    def __init__(self, vocab_size, cfg):
        super().__init__(vocab_size, cfg)
        self.field = VectorField(cfg.hidden_dim, cfg.embedding_dim, cfg.field_width, cfg.damping, cfg.gated)

    def solve(self, h, x, trajectory=False):
        return integrate(lambda y: self.field(y, x), h, self.cfg.steps_per_token,
                         self.cfg.integrator, trajectory)

    def transition(self, x, h):
        return self.solve(h, x)


class ResidualRNN(ConditionedNODE):
    """Exactly the conditioned field with Euler; weights can be copied without translation."""
    def solve(self, h, x, trajectory=False):
        return integrate(lambda y: self.field(y, x), h, self.cfg.steps_per_token, "euler", trajectory)

    @property
    def nfe_per_token(self):
        return self.cfg.steps_per_token
