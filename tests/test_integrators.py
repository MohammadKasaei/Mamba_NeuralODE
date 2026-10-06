import math
import pytest
import torch
from src.models.integrators import integrate, NFE_PER_STEP

@pytest.mark.parametrize("method,order", [("euler", 1), ("heun", 2), ("rk4", 4)])
def test_convergence(method, order):
    errors = []
    for k in (4, 8):
        out = integrate(lambda h: -h, torch.ones(2, 3, dtype=torch.float64), k, method)
        errors.append((out-math.exp(-1)).abs().max().item())
    assert errors[0]/errors[1] > 2**(order-0.25)

@pytest.mark.parametrize("method", ["euler", "heun", "rk4"])
def test_constant_control_and_nfe(method):
    calls = []
    def field(h):
        calls.append(1)
        return torch.full_like(h, 3)
    y, trajectory = integrate(field, torch.zeros(1, 2), 4, method, True)
    torch.testing.assert_close(y, torch.full_like(y, 3))
    assert len(calls) == 4*NFE_PER_STEP[method]
    assert len(trajectory) == 5


def test_gradient_and_invalid_arguments():
    h = torch.ones(2, requires_grad=True, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: integrate(lambda y: -y*y, x, 4, "rk4"), (h,))
    with pytest.raises(ValueError):
        integrate(lambda x: x, h, 0)
