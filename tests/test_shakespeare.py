import pytest
import torch
from src.config import Config
from src.data.tiny_shakespeare import ShakespeareData


def test_split_and_next_character(tmp_path):
    path = tmp_path/"chars.txt"
    path.write_text("abcd"*100 + "efgh"*20)
    data = ShakespeareData(Config(data_path=str(path)))
    assert len(data.splits["train"])+len(data.splits["val"]) == 480
    assert data.metadata["vocab"] == list("abcdefgh")
    x, y = data.batch(4, 16, 7, "val")
    assert torch.equal(x[:, 1:], y[:, :-1])
    x2, y2 = data.batch(4, 16, 7, "val")
    assert torch.equal(x, x2) and torch.equal(y, y2)
    with pytest.raises(ValueError):
        data.batch(4, 100, 7, "val")
