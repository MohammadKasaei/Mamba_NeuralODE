from .base import ODELM
from .integrators import integrate
from .vector_fields import DiagonalField

class StructuredNODE(ODELM):
    def __init__(self, vocab_size, cfg):
        super().__init__(vocab_size, cfg)
        self.field = DiagonalField(cfg.hidden_dim, cfg.embedding_dim, cfg.a_conditioned,
                                   cfg.b_conditioned, cfg.stable_a, cfg.structured_drive)

    def solve(self, h, x, trajectory=False):
        # Coefficients are constant within a token: evaluate the control maps once.
        a, b = self.field.coefficients(x)
        return integrate(lambda y: a*y+b, h, self.cfg.steps_per_token, self.cfg.integrator, trajectory)

    def transition(self, x, h):
        return self.solve(h, x)
