from dataclasses import replace
import csv
import json
import shutil
import pytest
import torch
from src.config import Config
from src.data.tiny_shakespeare import ShakespeareData
from src.data.synthetic import SyntheticData
from src.data.interventions import RemovedMemoryCue
from src.training import trainer


def test_language_reserved_test_partition(tmp_path):
    path = tmp_path/"unique.txt"
    path.write_text("".join(chr(0x400+i) for i in range(1000)))
    cfg = Config(data_path=str(path), language_train_fraction=0.8, language_val_fraction=0.1)
    data = ShakespeareData(cfg)
    assert [len(data.splits[s]) for s in ("train", "val", "test")] == [800, 100, 100]
    sets = [set(data.splits[s].tolist()) for s in ("train", "val", "test")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    x, y = data.batch(3, 20, 8, "test")
    assert set(x.flatten().tolist()) <= sets[2]
    assert set(y.flatten().tolist()) <= sets[2]
    assert torch.equal(x[:, 1:], y[:, :-1])
    with pytest.raises(ValueError, match="split fractions"):
        replace(cfg, language_val_fraction=0.3).validate()


@pytest.mark.parametrize("task", ["selective_copying", "induction"])
def test_memory_cue_removal_preserves_answers(task):
    cfg = Config(task=task)
    data = SyntheticData(cfg)
    x, y = data.batch(12, 64, 3, "val")
    changed, labels = RemovedMemoryCue(data, task).batch(12, 64, 3, "val")
    assert torch.equal(labels, y)
    if task == "selective_copying":
        assert not (changed == 1).any()
        assert torch.equal(changed[x != 1], x[x != 1])
    else:
        assert torch.equal(changed[:, -1], x[:, -1])
        assert ((changed != x).sum(1) == 1).all()
        assert not (changed[:, :-1] == changed[:, -1:]).any()


def test_early_stopping_and_resumed_patience(tmp_path, monkeypatch):
    cfg = Config(model="gru", embedding_dim=8, hidden_dim=12, vocab_symbols=4, recall_pairs=2,
                 batch_size=2, seq_len=12, train_steps=12, eval_every=1, eval_batches=1,
                 early_stopping_patience=2, early_stopping_min_steps=2,
                 warmup_steps=0, device="cpu", amp=False, output_dir=str(tmp_path/"full"))
    original_eval = trainer.evaluate
    def constant(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        result.update(loss=1.0)
        return result
    monkeypatch.setattr(trainer, "evaluate", constant)
    original_save = trainer.save_checkpoint
    captured = tmp_path/"at3.pt"
    def save(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[6] == 3 and str(args[0]).endswith("final.pt"):
            shutil.copyfile(args[0], captured)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    trainer.train(cfg)
    final = trainer.load_checkpoint(tmp_path/"full/final.pt")
    assert final["step"] == 4 and final["stopping_state"]["stale_checks"] == 2
    assert json.loads((tmp_path/"full/training_summary.json").read_text())["status"] == "early_stopped"
    trainer.train(replace(cfg, resume=str(captured), output_dir=str(tmp_path/"resumed")))
    resumed = trainer.load_checkpoint(tmp_path/"resumed/final.pt")
    assert resumed["step"] == final["step"]
    for key in final["model"]:
        assert torch.equal(final["model"][key], resumed["model"][key])
    # Resuming in the original directory must discard rows beyond the checkpoint.
    trainer.train(replace(cfg, resume=str(captured)))
    rows = list(csv.DictReader((tmp_path/"full/metrics.csv").open()))
    assert [int(row["step"]) for row in rows] == [1, 2, 3, 4]


def test_pending_and_nonfinite_runs_are_explicit_in_report(tmp_path):
    from src.analysis.all_tasks import report, TEST_SEED_OFFSET
    cfg = Config(model="gru", embedding_dim=8, hidden_dim=12, vocab_symbols=4, recall_pairs=2,
                 batch_size=2, seq_len=16, eval_batches=1, eval_lengths=[32])
    run = tmp_path/"run"
    entry = dict(name="recall_gru_s0", task=cfg.task, model=cfg.model, seed=0,
                 parameters=1000, run=str(run), config_values=cfg.to_dict())
    protocol = dict(tasks=[cfg.task], models=[cfg.model], seeds=[0], entries=[entry])
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    rows = report(tmp_path, plots=False)
    assert rows[0]["status"] == "planned" and "test_accuracy" not in rows[0]
    assert "0/1 planned runs" in (tmp_path/"summary/associative_recall/report.md").read_text()
    run.mkdir()
    (run/"test_results.json").write_text(json.dumps(dict(seed_offset=TEST_SEED_OFFSET, split="val", step=10,
        metrics=[dict(seq_len=16, finite=False, error="nonfinite"), dict(seq_len=32, finite=False, error="nonfinite")], memory_control=None)))
    rows = report(tmp_path, plots=False)
    assert rows[0]["tested"] and not rows[0]["test_finite"]
    assert "test_accuracy" not in rows[0]
    assert "Study incomplete" in (tmp_path/"summary/associative_recall/report.md").read_text()


def test_larger_training_batch_keeps_validation_samples_fixed():
    from src.models.factory import build_model
    cfg = Config(model="gru", embedding_dim=8, hidden_dim=12, vocab_symbols=4, recall_pairs=2,
                 batch_size=8, eval_batch_size=2, seq_len=16, eval_batches=2, device="cpu", amp=False)
    model = build_model(cfg, 12)
    data = SyntheticData(cfg)
    small = trainer.evaluate(model, data, cfg, torch.device("cpu"))
    large = trainer.evaluate(model, data, replace(cfg, batch_size=1024), torch.device("cpu"))
    assert small["examples"] == large["examples"] == 4
    assert small["loss"] == large["loss"]
    assert small["accuracy"] == large["accuracy"]


def test_long_context_microbatch_preserves_holdout_count(tmp_path):
    from scripts.eval_all_tasks import run
    cfg = Config(model="gru", embedding_dim=8, hidden_dim=12, vocab_symbols=4, recall_pairs=2,
                 batch_size=8, eval_batch_size=4, eval_token_budget=32,
                 seq_len=16, eval_batches=2, eval_lengths=[32], train_steps=1, eval_every=1,
                 device="cpu", amp=False, output_dir=str(tmp_path/"run"))
    trainer.train(cfg)
    result = run(tmp_path/"run/best.pt")
    assert [p["examples"] for p in result["metrics"]] == [8, 8]
    assert all(p["finite"] for p in result["metrics"])
    assert result["memory_control"]["examples"] == 8
