import pytest
import torch
from src.config import Config
from src.models.factory import MODEL_NAMES, build_model

@pytest.mark.parametrize("model", MODEL_NAMES)
def test_shapes_backward_and_causality(model):
    torch.manual_seed(1)
    cfg = Config(model=model, embedding_dim=8, hidden_dim=12, latent_dim=10,
                 field_width=16, transformer_heads=2, transformer_ff=16, steps_per_token=2)
    network = build_model(cfg, 20)
    tokens = torch.randint(20, (2, 8))
    logits = network(tokens)
    assert logits.shape == (2, 8, 20)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:]+1)%20
    torch.testing.assert_close(network(tokens)[:, :5], network(changed)[:, :5])
    if model != "transformer":
        first, state, _ = network(tokens[:, :4], return_stats=True)
        second, _, _ = network(tokens[:, 4:], state=state, return_stats=True)
        torch.testing.assert_close(torch.cat((first, second), 1), logits)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP test")
@pytest.mark.parametrize("model", [m for m in MODEL_NAMES if m not in ("gru", "lstm", "transformer")])
def test_amp_ode_memory_is_fp32(model, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = Config(model=model, embedding_dim=8, hidden_dim=12, latent_dim=10, field_width=16)
    network = build_model(cfg, 20).cuda()
    tokens = torch.randint(20, (2, 8), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits, state, _ = network(tokens, return_stats=True)
    assert network.memory(state).dtype == torch.float32
    assert torch.isfinite(logits).all()
