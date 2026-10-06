"""Independent-test analysis of the requested architecture/solver comparisons."""
from collections import defaultdict
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from src.config import Config
from src.data.factory import build_data
from src.training.trainer import batch_seed
from .plotting import read_csv, save


def label(cfg):
    if cfg["model"] == "conditioned":
        return f"conditioned_{cfg['integrator']}_k{cfg['steps_per_token']}"
    return cfg["model"]


def mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def report(root):
    root = Path(root)
    protocol = json.loads((root/"protocol.json").read_text())
    output = root/"summary"
    output.mkdir(exist_ok=True)
    import yaml
    rows, groups, extrapolation = [], defaultdict(list), defaultdict(list)
    for entry in protocol["entries"]:
        run = Path(entry["run"])
        if not (run/"extrapolation.json").exists():
            continue
        cfg = yaml.safe_load((run/"config.yaml").read_text())
        test = json.loads((run/"extrapolation.json").read_text())
        if test["seed_offset"] != protocol["test_seed_offset"]:
            raise ValueError(f"Wrong holdout stream: {run}")
        point = next(p for p in test["metrics"] if p["seq_len"] == cfg["seq_len"])
        if not point.get("finite", True):
            continue
        metrics = read_csv(run/"metrics.csv")
        metadata = json.loads((run/"metadata.json").read_text())
        bench = json.loads((run/"benchmark.json").read_text())
        key = label(cfg)
        row = dict(condition=key, seed=cfg["seed"], parameters=metadata["parameters"],
                   hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], field_width=cfg["field_width"],
                   embedding_dim=cfg["embedding_dim"], test_accuracy=point["accuracy"], test_loss=point["loss"], test_examples=point["supervised_tokens"],
                   query_changed_accuracy=test["query_ablation"]["accuracy"],
                   query_accuracy_drop=point["accuracy"]-test["query_ablation"]["accuracy"],
                   checkpoint_step=test["step"], attempted_updates=int(metrics[-1]["step"]),
                   skipped_updates=int(metrics[-1]["skipped_updates"]),
                   successful_updates=int(metrics[-1]["step"])-int(metrics[-1]["skipped_updates"]),
                   final_train_loss_mean50=float(np.mean([float(r["train_loss"]) for r in metrics[-50:]])),
                   training_seconds=float(metrics[-1]["training_seconds"]),
                   train_tokens_per_sec=bench["train_tokens_per_sec"], inference_tokens_per_sec=bench["inference_tokens_per_sec"],
                   nfe_per_token=metadata["nfe_per_token"], benchmark_peak_memory_bytes=bench["train_peak_memory_bytes"],
                   run=str(run))
        rows.append(row)
        groups[key].append(row)
        extrapolation[key].append((cfg["seed"], test["metrics"]))
    if not rows:
        raise ValueError("No focused test observations")
    with (output/"test_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    baselines = []
    cfg = Config(**protocol["config"])
    lengths = sorted(set([cfg.seq_len]+list(cfg.eval_lengths)))
    for seed in protocol["seeds"]:
        seeded = replace(cfg, seed=seed)
        data = build_data(seeded)
        for length in lengths:
            modal, recent, oracle, total = 0, 0, 0, 0
            for i in range(cfg.eval_batches):
                x, y = data.batch(cfg.batch_size, length, batch_seed(seeded, i, validation=True, length=length)+protocol["test_seed_offset"], "val")
                for tokens, target in zip(x, y):
                    values = tokens[tokens >= 4+cfg.vocab_symbols]
                    modal += int(torch.bincount(values, minlength=data.vocab_size).argmax() == target[-1])
                    recent += int(values[-1] == target[-1])
                    key_position = (tokens[:-2] == tokens[-2]).nonzero().flatten()
                    oracle += int(tokens[key_position[0]+1] == target[-1])
                    total += 1
            assert oracle == total, "Task oracle must recover every answer"
            baselines.append(dict(seed=seed, seq_len=length, examples=total,
                                  modal_value_accuracy=modal/total, recent_value_accuracy=recent/total,
                                  oracle_accuracy=oracle/total, uniform_guess_accuracy=1/cfg.vocab_symbols))
    with (output/"shortcut_baselines.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(baselines[0])); writer.writeheader(); writer.writerows(baselines)
    aggregated = []
    for key, entries in groups.items():
        row = dict(condition=key, seeds=len(entries))
        for metric in ("test_accuracy", "test_loss", "query_changed_accuracy", "query_accuracy_drop", "final_train_loss_mean50", "training_seconds", "train_tokens_per_sec", "inference_tokens_per_sec", "skipped_updates"):
            mean, sd = mean_sd([r[metric] for r in entries]); row[f"{metric}_mean"], row[f"{metric}_sd"] = mean, sd
        row.update(parameters_min=min(r["parameters"] for r in entries), parameters_max=max(r["parameters"] for r in entries),
                   nfe_per_token=entries[0]["nfe_per_token"])
        aggregated.append(row)
    with (output/"seed_aggregates.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregated[0])); writer.writeheader(); writer.writerows(aggregated)
    contrasts = [("conditioned_euler_k1", "autonomous"), ("conditioned_euler_k1", "structured"),
                 ("conditioned_euler_k1", "gru"), ("conditioned_euler_k4", "conditioned_euler_k1"),
                 ("conditioned_heun_k4", "conditioned_euler_k1"), ("conditioned_heun_k4", "conditioned_euler_k4")]
    paired = []
    for left, right in contrasts:
        lm = {r["seed"]: r for r in groups.get(left, [])}; rm = {r["seed"]: r for r in groups.get(right, [])}
        seeds = sorted(lm.keys() & rm.keys())
        if not seeds:
            continue
        diffs = [100*(lm[s]["test_accuracy"]-rm[s]["test_accuracy"]) for s in seeds]
        mean, sd = mean_sd(diffs)
        margin = 4.302652729911275*sd/math.sqrt(3) if len(seeds) == 3 else None
        paired.append(dict(left=left, right=right, paired_seeds=len(seeds), mean_difference_pp=mean,
                           sd_difference_pp=sd, ci95_low_pp=mean-margin if margin is not None else None,
                           ci95_high_pp=mean+margin if margin is not None else None, seed_differences_pp=diffs))
    (output/"paired_contrasts.json").write_text(json.dumps(paired, indent=2))
    base_at_train = [b for b in baselines if b["seq_len"] == cfg.seq_len]
    modal_mean, modal_sd = mean_sd([b["modal_value_accuracy"] for b in base_at_train])
    recent_mean, recent_sd = mean_sd([b["recent_value_accuracy"] for b in base_at_train])
    order = ["gru", "autonomous", "conditioned_euler_k1", "structured", "conditioned_euler_k4", "conditioned_heun_k4"]
    order = [key for key in order if key in groups]
    fig, ax = plt.subplots(figsize=(10, 4))
    means = [100*np.mean([r["test_accuracy"] for r in groups[k]]) for k in order]
    stds = [100*mean_sd([r["test_accuracy"] for r in groups[k]])[1] for k in order]
    ax.bar(range(len(order)), means, yerr=stds, capsize=4)
    ax.axhline(100*modal_mean, label="Most frequent context value", color="tab:red", linestyle="--")
    ax.axhline(100*recent_mean, label="Most recent context value", color="tab:orange", linestyle="--")
    ax.set_xticks(range(len(order)), order, rotation=25, ha="right")
    ax.set(ylabel="Independent test accuracy (%)", title="Mean ± sample SD across three seeds")
    ax.legend(fontsize=8); save(fig, output/"focused_test_accuracy.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    for key in order:
        points = defaultdict(list)
        for _, entries in extrapolation[key]:
            for point in entries:
                if point.get("finite", True): points[point["seq_len"]].append(point["accuracy"]*100)
        xs = sorted(points); ys = np.array([np.mean(points[x]) for x in xs])
        sd = np.array([mean_sd(points[x])[1] for x in xs])
        ax.plot(xs, ys, marker="o", label=key); ax.fill_between(xs, ys-sd, ys+sd, alpha=0.12)
    bm = [np.mean([b["modal_value_accuracy"]*100 for b in baselines if b["seq_len"] == x]) for x in lengths]
    ax.plot(lengths, bm, "k--", label="Most frequent context value")
    ax.set_xscale("log", base=2); ax.set(xlabel="Test sequence length", ylabel="Independent test accuracy (%)")
    ax.legend(fontsize=7); save(fig, output/"focused_length_extrapolation.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in order:
        entries = groups[key]
        axes[0].scatter([np.mean([r["training_seconds"] for r in entries])], [100*np.mean([r["test_accuracy"] for r in entries])], label=key)
        axes[1].scatter([np.mean([r["train_tokens_per_sec"] for r in entries])], [100*np.mean([r["test_accuracy"] for r in entries])], label=key)
    axes[0].set(xlabel="Mean measured training seconds", ylabel="Test accuracy (%)")
    axes[1].set(xlabel="Isolated training benchmark tokens/s", ylabel="Test accuracy (%)")
    axes[1].legend(fontsize=7); save(fig, output/"focused_quality_compute.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in order:
        train, validation = defaultdict(list), defaultdict(list)
        for entry in groups[key]:
            for point in read_csv(Path(entry["run"])/"metrics.csv"):
                step = int(point["step"])
                train[step].append(float(point["train_loss"]))
                if point["val_loss"]: validation[step].append(float(point["val_loss"]))
        steps = sorted(train); means = [np.mean(train[s]) for s in steps]
        if len(means) >= 50:
            axes[0].plot(steps[49:], np.convolve(means, np.ones(50)/50, mode="valid"), label=key)
        xs = sorted(validation); ys = np.array([np.mean(validation[x]) for x in xs]); sd = np.array([mean_sd(validation[x])[1] for x in xs])
        axes[1].plot(xs, ys, label=key); axes[1].fill_between(xs, ys-sd, ys+sd, alpha=0.12)
    axes[0].set(xlabel="Attempted update", ylabel="Training CE, 50-step moving average")
    axes[1].set(xlabel="Attempted update", ylabel="Validation CE, mean ± sample SD")
    axes[1].legend(fontsize=7); save(fig, output/"focused_learning_curves.png")
    lines = ["# Three-seed recall follow-up", "", f"{len(rows)}/{len(protocol['entries'])} runs have independent-test observations.",
             f"Training: length {cfg.seq_len}, {cfg.train_steps} attempted updates, global batch {protocol['effective_batch']}, embedding {cfg.embedding_dim}, reference state {cfg.hidden_dim}, {cfg.recall_pairs} pairs, {cfg.vocab_symbols} values.",
             f"Same optimizer/schedule/data streams within each seed; best checkpoint selected by validation CE. Test uses a separate synthetic RNG stream. Every length has {protocol['examples_per_test_length']} examples per seed.",
             "Capacity is matched; structured state size differs, while conditioned and autonomous state/latent sizes match. Native GRU uses AMP cell precision, ODE accumulators use FP32. These are compact, fixed-budget results rather than established convergence.", "",
             "| Condition | Seeds | Test accuracy, mean ± SD | Test CE | Train CE, final 50-step mean | Training seconds | Isolated tokens/s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    by_key = {r["condition"]: r for r in aggregated}
    for key in order:
        r = by_key[key]
        lines.append(f"| {key} | {r['seeds']} | {100*r['test_accuracy_mean']:.2f}% ± {100*r['test_accuracy_sd']:.2f} pp | {r['test_loss_mean']:.4f} | {r['final_train_loss_mean50_mean']:.4f} | {r['training_seconds_mean']:.1f} | {r['train_tokens_per_sec_mean']:,.0f} |")
    lines += ["", f"Uniform guessing: {100/cfg.vocab_symbols:.2f}%. Most frequent context value: {100*modal_mean:.2f}% ± {100*modal_sd:.2f} pp. Most recent context value: {100*recent_mean:.2f}% ± {100*recent_sd:.2f} pp. The lookup oracle achieves 100%.",
              "Accuracy above uniform guessing alone does not demonstrate key-specific recall. The frequent-value baseline uses no query key.", "",
              "| Paired contrast (left minus right) | Difference, pp | Exploratory 95% t interval, pp |",
              "|---|---:|---:|"]
    for p in paired:
        ci = f"[{p['ci95_low_pp']:.2f}, {p['ci95_high_pp']:.2f}]" if p["ci95_low_pp"] is not None else "unavailable"
        lines.append(f"| {p['left']} − {p['right']} | {p['mean_difference_pp']:.2f} | {ci} |")
    lines += ["", "| Query-key intervention | Accuracy with changed query | Original minus changed, pp |", "|---|---:|---:|"]
    for key in order:
        r = by_key[key]
        lines.append(f"| {key} | {100*r['query_changed_accuracy_mean']:.2f}% | {100*r['query_accuracy_drop_mean']:.2f} ± {100*r['query_accuracy_drop_sd']:.2f} |")
    lines += ["", "The query is replaced by a different key present in the same context, while labels stay tied to the original key. A positive accuracy drop supports use of the correct key; context-frequency predictors are unaffected. Errors due to repeated values remain possible.", "", "Intervals use paired differences across three seeds (Student t, df=2). They are unadjusted exploratory intervals across six comparisons, and have low power. They do not establish equivalence when they include zero.",
              "AMP skipped-update counts, exact capacity, per-seed results and compute are in test_summary.csv. Finite length points are in each run's extrapolation.json; failures remain explicit. Training times were measured while independent jobs used separate GPUs; isolated benchmark measurements were run sequentially on one common GPU after all training completed.", ""]
    (output/"focused_report.md").write_text("\n".join(lines))
    print(f"Focused report: {output/'focused_report.md'}", flush=True)
    return rows
