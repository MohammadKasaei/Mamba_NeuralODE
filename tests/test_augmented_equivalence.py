from dataclasses import replace
import pytest
import torch
from src.config import Config
from src.models.factory import build_model

@pytest.mark.parametrize("method", ["euler", "heun", "rk4"])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_augmented_trajectory_and_gradients(method, k):
    torch.manual_seed(7)
    cfg = Config(embedding_dim=8, hidden_dim=12, field_width=16,
                 integrator=method, steps_per_token=k)
    direct = build_model(cfg, 20)
    augmented = build_model(replace(cfg, model="augmented"), 20)
    augmented.load_state_dict(direct.state_dict())
    h = torch.randn(3, 12, requires_grad=True)
    x = torch.randn(3, 8, requires_grad=True)
    hd, td = direct.solve(h, x, True)
    za, ta = augmented.solve_augmented(h, x, True)
    for d, a in zip(td, ta):
        assert (d-a[:, :12]).abs().max().item() < 1e-6
        assert torch.equal(a[:, 12:], x)
    gd = torch.autograd.grad(hd.sum(), (h, x), retain_graph=True)
    ga = torch.autograd.grad(za[:, :12].sum(), (h, x))
    for d, a in zip(gd, ga):
        torch.testing.assert_close(d, a, atol=1e-6, rtol=1e-6)

@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_residual_equals_euler(k):
    cfg = Config(embedding_dim=8, hidden_dim=12, field_width=16, steps_per_token=k)
    node = build_model(cfg, 20)
    residual = build_model(replace(cfg, model="residual"), 20)
    residual.load_state_dict(node.state_dict())
    x = torch.randint(20, (2, 5))
    assert torch.equal(node(x), residual(x))
