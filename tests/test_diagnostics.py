import csv
import json
import torch
from src.config import Config
from src.analysis.dynamics import diagnose
from src.models.factory import build_model


def test_structured_discrete_stability_is_not_continuous_stability(tmp_path):
    cfg = Config(model="structured", hidden_dim=4, embedding_dim=4, field_width=4,
                 a_conditioned=False, b_conditioned=False, stable_a=True, steps_per_token=1)
    model = build_model(cfg, 8)
    with torch.no_grad():
        model.field.a_bias.fill_(4)  # -softplus(4) < -4; Euler multiplier below -3.
    report = diagnose(model, torch.tensor([[1, 2, 3]]), tmp_path)
    assert report["negative_fraction"] == 1
    assert report["discretely_stable_fraction"] == 0
    assert report["max_discrete_amplification"] > 3
    rows = list(csv.DictReader((tmp_path/"internal_trajectory.csv").open()))
    assert len(rows) == 3*2
    trajectory = torch.load(tmp_path/"internal_states.pt", weights_only=True)
    assert trajectory.shape == (1, 3, 2, 4)
    assert json.loads((tmp_path/"diagnostics.json").read_text())["finite_trajectory"]
