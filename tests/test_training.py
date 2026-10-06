from dataclasses import replace
import shutil
import torch
from src.config import Config
from src.training import trainer
from src.models.factory import build_model, match_capacity, parameter_count


def test_checkpoint_resume_is_exact(tmp_path, monkeypatch):
    cfg = Config(model="transformer", embedding_dim=8, hidden_dim=12, latent_dim=12, field_width=12,
                 transformer_ff=16, transformer_heads=2, dropout=0.1,
                 vocab_symbols=4, recall_pairs=2, batch_size=2, seq_len=12,
                 train_steps=4, eval_every=2, eval_batches=1, warmup_steps=0,
                 device="cpu", amp=False, output_dir=str(tmp_path/"full"))
    original = trainer.save_checkpoint
    captured = tmp_path/"at2.pt"
    def save(*args, **kwargs):
        original(*args, **kwargs)
        if args[6] == 2 and str(args[0]).endswith("final.pt"):
            shutil.copyfile(args[0], captured)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    trainer.train(cfg)
    final = trainer.load_checkpoint(tmp_path/"full/final.pt")
    trainer.train(replace(cfg, resume=str(captured), output_dir=str(tmp_path/"resumed")))
    resumed = trainer.load_checkpoint(tmp_path/"resumed/final.pt")
    for key in final["model"]:
        assert torch.equal(final["model"][key], resumed["model"][key])
    assert final["step"] == resumed["step"] == 4


def test_gradient_accumulation_partition(tmp_path):
    cfg = Config(model="gru", embedding_dim=8, hidden_dim=12, vocab_symbols=4, recall_pairs=2,
                 batch_size=4, seq_len=12, train_steps=2, eval_every=2, eval_batches=1,
                 warmup_steps=0, device="cpu", amp=False, output_dir=str(tmp_path/"batch4"))
    trainer.train(cfg)
    trainer.train(replace(cfg, batch_size=2, accumulation_steps=2, output_dir=str(tmp_path/"batch2")))
    first = trainer.load_checkpoint(tmp_path/"batch4/final.pt")
    second = trainer.load_checkpoint(tmp_path/"batch2/final.pt")
    for key in first["model"]:
        torch.testing.assert_close(first["model"][key], second["model"][key], atol=1e-6, rtol=1e-5)


def test_matching_does_not_change_embeddings():
    for name in ("gru", "lstm", "autonomous", "conditioned", "projected_conditioned", "structured", "transformer"):
        cfg = Config(model=name, embedding_dim=16, hidden_dim=32, latent_dim=32, field_width=32,
                     transformer_ff=64, match_parameters=12000)
        matched = match_capacity(cfg, 20)
        assert matched.embedding_dim == cfg.embedding_dim
        assert abs(parameter_count(build_model(matched, 20))/12000 - 1) < 0.1


def test_independent_synthetic_holdout_seed_offset():
    from src.data.factory import build_data
    cfg = Config(model="gru", embedding_dim=8, hidden_dim=12, vocab_symbols=4, recall_pairs=2,
                 batch_size=2, seq_len=12, eval_batches=2, device="cpu", amp=False)
    data = build_data(cfg)
    seeds = []
    class RecordingData:
        def batch(self, batch_size, length, seed, split):
            seeds.append(seed)
            return data.batch(batch_size, length, seed, split)
    network = build_model(cfg, data.vocab_size)
    trainer.evaluate(network, RecordingData(), cfg, torch.device("cpu"))
    first = seeds.copy(); seeds.clear()
    trainer.evaluate(network, RecordingData(), cfg, torch.device("cpu"), seed_offset=1000000000000)
    assert seeds == [seed+1000000000000 for seed in first]
    assert network.training
