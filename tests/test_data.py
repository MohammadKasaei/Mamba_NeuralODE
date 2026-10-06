import pytest
import torch
from src.config import Config
from src.data.synthetic import SyntheticData, TASKS
from src.models.vector_fields import DiagonalField

@pytest.mark.parametrize("task", TASKS)
def test_data(task):
    cfg = Config(task=task, copy_items=3, recall_pairs=4, vocab_symbols=8)
    data = SyntheticData(cfg)
    x, y = data.batch(16, 32, 42)
    x2, y2 = data.batch(16, 32, 42)
    assert torch.equal(x, x2) and torch.equal(y, y2)
    assert x.shape == y.shape == (16, 32)
    assert (y != -100).sum().item() == 16*(3 if task == "selective_copying" else 1)
    for a, b in zip(x, y):
        if task == "selective_copying":
            selected = a[(a == 1).nonzero().flatten()+1]
            assert torch.equal(selected, b[b != -100])
        elif task == "associative_recall":
            loc = (a[:-2] == a[-2]).nonzero().flatten()
            assert len(loc) == 1 and a[loc[0]+1] == b[-1]
        else:
            loc = (a[:-1] == a[-1]).nonzero().flatten()
            assert len(loc) == 1 and a[loc[0]+1] == b[-1]

@pytest.mark.parametrize("ac", [False, True])
@pytest.mark.parametrize("bc", [False, True])
@pytest.mark.parametrize("drive", ["bias", "matrix"])
def test_structured_coefficients(ac, bc, drive):
    field = DiagonalField(12, 8, ac, bc, True, drive)
    x = torch.randn(3, 8)
    a, b = field.coefficients(x)
    assert a.shape == b.shape == (3, 12)
    assert (a < 0).all()
    if not ac:
        torch.testing.assert_close(a[0], a[1])
    if not bc and drive == "bias":
        torch.testing.assert_close(b[0], b[1])


def test_query_intervention_preserves_context_and_labels():
    from src.data.interventions import ChangedRecallQuery
    cfg = Config(task="associative_recall", vocab_symbols=8, recall_pairs=4)
    data = SyntheticData(cfg)
    x, y = data.batch(16, 32, 123)
    changed, targets = ChangedRecallQuery(data, cfg.vocab_symbols).batch(16, 32, 123)
    assert torch.equal(x[:, :-2], changed[:, :-2])
    assert torch.equal(x[:, -1], changed[:, -1])
    assert torch.equal(y, targets)
    assert (x[:, -2] != changed[:, -2]).all()
    for row in changed:
        assert (row[:-2] == row[-2]).sum() == 1
