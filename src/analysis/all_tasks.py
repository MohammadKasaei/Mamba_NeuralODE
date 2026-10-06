"""Task-specific held-out reports, including incomplete runs and shortcut controls."""
from collections import defaultdict
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import Config
from src.data.factory import build_data
from src.models.factory import MODEL_NAMES
from src.training.trainer import batch_seed
from .plotting import read_csv, save, dynamics_plots

TEST_SEED_OFFSET = 1000000000000
TASK_DESCRIPTIONS = {
    "selective_copying": "Recall four marked symbols in order after a COPY cue, ignoring unmarked distractors. Answer-token loss only; earlier answer tokens are teacher-forced. Whole-sequence accuracy requires all four answers to be correct.",
    "associative_recall": "Remember eight unique key/value pairs scattered through padding and retrieve the value of the final queried key. Values may repeat. Answer-token loss only.",
    "induction": "A random stream has exactly one earlier occurrence of the query symbol. Predict the symbol that followed that earlier occurrence when the query reappears at the end. Answer-token loss only.",
    "shakespeare": "Predict every next character in Tiny Shakespeare. Contiguous 80% training, 10% validation and 10% reserved test text; sampled windows never cross boundaries. Long windows begin with fresh state and score every character, so this is window-length evaluation rather than a retrieval-distance test.",
}


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    values = [float(v) for v in values if v is not None and v != "" and math.isfinite(float(v))]
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0) if values else (None, None)


def baselines(cfg, seeds):
    """Evaluate simple controls on exactly the same sampled holdout batches."""
    rows = []
    data = build_data(cfg)
    if cfg.task == "shakespeare":
        train = data.splits["train"]
        unigram = torch.bincount(train, minlength=data.vocab_size).double()+1
        unigram /= unigram.sum()
        counts = torch.bincount(train[:-1]*data.vocab_size+train[1:], minlength=data.vocab_size**2).reshape(data.vocab_size, -1).double()+1
        bigram = counts/counts.sum(1, keepdim=True)
    for seed in seeds:
        seeded = replace(cfg, seed=seed)
        for length in sorted(set([cfg.seq_len, *cfg.eval_lengths])):
            batch = min(cfg.evaluation_batch_size, max(1, cfg.eval_token_budget//length))
            batches = cfg.evaluation_batch_size*cfg.eval_batches//batch
            totals = defaultdict(float)
            for i in range(batches):
                x, y = data.batch(batch, length, batch_seed(seeded, i, validation=True, length=length)+TEST_SEED_OFFSET,
                                  "test" if cfg.task == "shakespeare" else "val")
                if cfg.task == "shakespeare":
                    totals["unigram_loss"] += float(-unigram[y].log().sum())
                    totals["bigram_loss"] += float(-bigram[x, y].log().sum())
                    totals["unigram_accuracy"] += int((unigram.argmax() == y).sum())
                    totals["bigram_accuracy"] += int((bigram.argmax(1)[x] == y).sum())
                    totals["count"] += y.numel()
                    continue
                for tokens, labels in zip(x, y):
                    targets = labels[labels != -100]
                    if cfg.task == "associative_recall":
                        context = tokens[:-2]
                        values = context[context >= 4+cfg.vocab_symbols]
                        oracle = tokens[(context == tokens[-2]).nonzero().flatten()+1]
                        recent = values[-1:]
                    elif cfg.task == "selective_copying":
                        end = (tokens == 3).nonzero().flatten().item()
                        context = tokens[:end]
                        values = context[context >= 4]
                        oracle = context[(context == 1).nonzero().flatten()+1]
                        recent = values[-cfg.copy_items:]
                    else:
                        context = tokens[:-1]
                        values = context
                        oracle = tokens[(context == tokens[-1]).nonzero().flatten()+1]
                        recent = values[-1:]
                    assert torch.equal(oracle, targets), "Task oracle must recover the labels"
                    modal = torch.bincount(values, minlength=data.vocab_size).argmax()
                    totals["frequency_accuracy"] += int((modal == targets).sum())
                    totals["recent_accuracy"] += int((recent == targets).sum())
                    totals["oracle_accuracy"] += len(targets)
                    totals["count"] += len(targets)
            count = totals.pop("count")
            row = dict(seed=seed, seq_len=length, supervised_tokens=int(count), **{k: v/count for k, v in totals.items()})
            if cfg.task == "shakespeare":
                row.update(unigram_bpc=row["unigram_loss"]/math.log(2), bigram_bpc=row["bigram_loss"]/math.log(2))
            else:
                row["uniform_guess_accuracy"] = 1/cfg.vocab_symbols
            rows.append(row)
    return rows


def report(root, plots=True):
    root = Path(root)
    protocol = json.loads((root/"protocol.json").read_text())
    output = root/"summary"
    output.mkdir(exist_ok=True)
    statuses = json.loads((root/"progress.json").read_text()) if (root/"progress.json").exists() else {}
    rows, lengths, diagnostics = [], [], []
    for entry in protocol["entries"]:
        run = Path(entry["run"])
        row = dict(task=entry["task"], model=entry["model"], seed=entry["seed"], run=str(run),
                   status=statuses.get(entry["name"], {}).get("status", "planned"), parameters=entry["parameters"])
        if (run/"training_summary.json").exists():
            row.update(json.loads((run/"training_summary.json").read_text()))
            row.pop("stopping_state", None)
        if (run/"failure.json").exists():
            row.update(status="failed", **json.loads((run/"failure.json").read_text()))
        metrics = read_csv(run/"metrics.csv")
        if metrics:
            val = [p for p in metrics if p["val_loss"]]
            row.update(final_train_loss=float(np.mean([float(p["train_loss"]) for p in metrics[-100:]])),
                       peak_memory_bytes=max(int(p["peak_memory_bytes"]) for p in metrics),
                       observed_train_tokens_per_sec=float(np.median([float(p["train_tokens_per_sec"]) for p in metrics[-100:]])))
            if val:
                best = min(val, key=lambda p: float(p["val_loss"]))
                row.update(best_validation_loss=float(best["val_loss"]), checkpoint_step=int(best["step"]),
                           final_validation_loss=float(val[-1]["val_loss"]))
        test_path = run/"test_results.json"
        if test_path.exists():
            test = json.loads(test_path.read_text())
            cfg = Config(**entry["config_values"])
            expected_split = "test" if cfg.task == "shakespeare" else "val"
            if test["seed_offset"] != TEST_SEED_OFFSET or test["split"] != expected_split:
                raise ValueError(f"Incorrect independent holdout: {run}")
            for point in test["metrics"]:
                lengths.append(dict(task=cfg.task, model=cfg.model, seed=cfg.seed, **point))
            point = next(p for p in test["metrics"] if p["seq_len"] == cfg.seq_len)
            row.update(tested=True, test_finite=point["finite"], checkpoint_step=test["step"])
            if point["finite"]:
                row.update(test_loss=point["loss"], test_accuracy=point["accuracy"], test_bpc=point["bpc"],
                           test_perplexity=point["perplexity"], exact_sequence_accuracy=point["exact_sequence_accuracy"],
                           test_examples=point["examples"], test_supervised_tokens=point["supervised_tokens"])
                control = test["memory_control"]
                if control and control.get("finite"):
                    row.update(memory_control_accuracy=control["accuracy"], memory_control_drop=point["accuracy"]-control["accuracy"])
            longest = max(test["metrics"], key=lambda p: p["seq_len"])
            row.update(longest_test_length=longest["seq_len"], longest_finite=longest["finite"])
            if longest["finite"]:
                row.update(longest_accuracy=longest["accuracy"], longest_bpc=longest["bpc"], longest_hidden_norm=longest["hidden_norm"])
        if (run/"benchmark.json").exists():
            benchmark = json.loads((run/"benchmark.json").read_text())
            for key in ("train_tokens_per_sec", "inference_tokens_per_sec", "nfe_per_token", "train_peak_memory_bytes",
                        "forward_flops_per_token_estimate", "train_ms_per_token", "inference_ms_per_token", "train_skipped_updates"):
                if key in benchmark:
                    row[key] = benchmark[key]
        if (run/"dynamics/diagnostics.json").exists():
            diagnostics.append(dict(task=entry["task"], model=entry["model"], seed=entry["seed"],
                                    **json.loads((run/"dynamics/diagnostics.json").read_text())))
        rows.append(row)
    write_csv(output/"all_runs.csv", rows)
    write_csv(output/"length_results.csv", lengths)
    write_csv(output/"dynamics_summary.csv", diagnostics)
    index = ["# All-task training study", "", f"Planned: {len(rows)} runs; independent-test results: {sum(bool(r.get('tested')) for r in rows)}; isolated benchmarks: {sum('train_tokens_per_sec' in r for r in rows)}.",
             "", "Reports include pending and failed runs explicitly. Mean ± sample SD describes variation across seeds, not a confidence interval. Validation selects checkpoints; test data never selects checkpoints or determines stopping.", ""]
    for task in protocol["tasks"]:
        folder = output/task
        folder.mkdir(exist_ok=True)
        selected = [r for r in rows if r["task"] == task]
        tested = [r for r in selected if r.get("test_finite")]
        cfg = Config(**next(e for e in protocol["entries"] if e["task"] == task)["config_values"])
        base_path = folder/"baselines.csv"
        if not base_path.exists():
            write_csv(base_path, baselines(cfg, protocol["seeds"]))
        controls = read_csv(base_path)
        write_csv(folder/"per_seed.csv", selected)
        groups = defaultdict(list)
        for row in tested:
            groups[row["model"]].append(row)
        aggregates = []
        for model in MODEL_NAMES:
            if model not in groups:
                continue
            runs = groups[model]
            aggregate = dict(model=model, seeds=len(runs))
            for metric in ("test_loss", "test_accuracy", "test_bpc", "test_perplexity", "exact_sequence_accuracy", "longest_accuracy", "longest_bpc",
                           "memory_control_accuracy", "memory_control_drop", "attempted_updates", "skipped_updates", "checkpoint_step", "training_seconds",
                           "final_train_loss", "train_tokens_per_sec", "inference_tokens_per_sec", "observed_train_tokens_per_sec", "peak_memory_bytes"):
                mean, sd = mean_sd([r.get(metric) for r in runs])
                aggregate.update({metric+"_mean": mean, metric+"_sd": sd})
            aggregates.append(aggregate)
        write_csv(folder/"seed_aggregates.csv", aggregates)
        paired = []
        for left, right in (("conditioned", "autonomous"), ("projected_conditioned", "autonomous"), ("conditioned", "structured"),
                            ("conditioned", "gru"), ("conditioned", "transformer"), ("conditioned", "residual"), ("conditioned", "augmented")):
            lm = {r["seed"]: r for r in groups.get(left, [])}; rm = {r["seed"]: r for r in groups.get(right, [])}
            seeds = sorted(lm.keys() & rm.keys())
            metric = "test_bpc" if task == "shakespeare" else "test_accuracy"
            scale = 1 if task == "shakespeare" else 100
            differences = [scale*(lm[s][metric]-rm[s][metric]) for s in seeds]
            if not differences:
                continue
            mean, sd = mean_sd(differences)
            margin = 4.302652729911275*sd/math.sqrt(3) if len(seeds) == 3 else None
            paired.append(dict(left=left, right=right, seeds=len(seeds), metric="BPC" if task == "shakespeare" else "accuracy pp", difference=mean,
                               ci95_low=mean-margin if margin is not None else None, ci95_high=mean+margin if margin is not None else None))
        (folder/"paired_contrasts.json").write_text(json.dumps(paired, indent=2))
        expected = len(selected)
        text = [f"# {task.replace('_', ' ').title()}", "", TASK_DESCRIPTIONS[task], "",
                f"Independent finite results: {len(tested)}/{expected} planned runs. Training length {cfg.seq_len}, embedding {cfg.embedding_dim}, reference state/latent {cfg.hidden_dim}; parameter matching within {100*cfg.match_tolerance:g}% of the GRU reference. Actual state/field widths are in saved configs.",
                f"AdamW LR {cfg.learning_rate}; linear warmup {cfg.warmup_steps} then cosine over the {cfg.train_steps:,}-update ceiling. Stop after {cfg.early_stopping_patience} validation checks without a CE improvement of {cfg.early_stopping_min_delta}, starting at update {cfg.early_stopping_min_steps}; validate every {cfg.eval_every} attempts on a fixed stream. Early stopping indicates a plateau under this optimizer and criterion, not global convergence.",
                f"Global batch {cfg.batch_size*cfg.accumulation_steps}; evaluation batch {cfg.evaluation_batch_size}; {cfg.evaluation_batch_size*cfg.eval_batches} held-out examples per seed and length. Models use independent single-GPU jobs; data streams and embedding initialization are paired within each seed.", "",
                "| Model | Seeds | Test accuracy | Test CE | Test BPC | Perplexity | Updates, mean | Best step, mean |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for a in aggregates:
            text.append(f"| {a['model']} | {a['seeds']} | {100*a['test_accuracy_mean']:.2f}% ± {100*a['test_accuracy_sd']:.2f} pp | {a['test_loss_mean']:.4f} ± {a['test_loss_sd']:.4f} | " +
                        (f"{a['test_bpc_mean']:.4f} | {a['test_perplexity_mean']:.2f}" if task == "shakespeare" else "— | —")+
                        f" | {a['attempted_updates_mean']:.0f} | {a['checkpoint_step_mean']:.0f} |")
        if task != "shakespeare":
            base = [r for r in controls if int(r["seq_len"]) == cfg.seq_len]
            freq, freq_sd = mean_sd([r["frequency_accuracy"] for r in base])
            recent, _ = mean_sd([r["recent_accuracy"] for r in base])
            text += ["", f"Controls: uniform guessing {100/cfg.vocab_symbols:.2f}%; context frequency {100*freq:.2f}% ± {100*freq_sd:.2f} pp; recent-value/symbol rule {100*recent:.2f}%; oracle 100%.",
                     "Above-chance accuracy alone does not establish the intended memory mechanism. Controls use the original answers. Cue removal changes the input distribution and measures sensitivity, not a causal proof of successful memory.", "",
                     "| Model | Whole-sequence accuracy | Accuracy after cue change | Accuracy drop, pp | Longest-context accuracy |",
                     "|---|---:|---:|---:|---:|"]
            for a in aggregates:
                control = a["memory_control_accuracy_mean"]
                drop = a["memory_control_drop_mean"]
                long = a["longest_accuracy_mean"]
                text.append(f"| {a['model']} | {100*a['exact_sequence_accuracy_mean']:.2f}% | {100*control:.2f}% | {100*drop:.2f} ± {100*a['memory_control_drop_sd']:.2f} | {100*long:.2f}% |" if control is not None and long is not None else f"| {a['model']} | — | — | — | nonfinite/missing |")
        else:
            base = [r for r in controls if int(r["seq_len"]) == cfg.seq_len]
            unigram, _ = mean_sd([r["unigram_bpc"] for r in base]); bigram, _ = mean_sd([r["bigram_bpc"] for r in base])
            text += ["", f"Training-text baselines on reserved test windows: unigram {unigram:.4f} BPC; add-one-smoothed bigram {bigram:.4f} BPC.", "",
                     "| Model | Longest-context test BPC | Training CE, final 100-step mean | Final / best validation CE |", "|---|---:|---:|---:|"]
            for a in aggregates:
                final, _ = mean_sd([r["final_validation_loss"] for r in groups[a["model"]]])
                best, _ = mean_sd([r["best_validation_loss"] for r in groups[a["model"]]])
                longest = a["longest_bpc_mean"]
                text.append(f"| {a['model']} | {longest:.4f} | {a['final_train_loss_mean']:.4f} | {final:.4f} / {best:.4f} |" if longest is not None else f"| {a['model']} | nonfinite | — | — |")
        text += ["", "| Model | Training seconds | Isolated train tokens/s | Isolated inference tokens/s | Peak training MiB | AMP skipped attempts |", "|---|---:|---:|---:|---:|---:|"]
        for a in aggregates:
            train = f"{a['train_tokens_per_sec_mean']:,.0f}" if a["train_tokens_per_sec_mean"] is not None else "pending"
            inference = f"{a['inference_tokens_per_sec_mean']:,.0f}" if a["inference_tokens_per_sec_mean"] is not None else "pending"
            text.append(f"| {a['model']} | {a['training_seconds_mean']:.1f} | {train} | {inference} | {a['peak_memory_bytes_mean']/2**20:.1f} | {a['skipped_updates_mean']:.1f} |")
        text += ["", "| Paired contrast, left minus right | Difference | Exploratory 95% interval |", "|---|---:|---:|"]
        for p in paired:
            interval = f"[{p['ci95_low']:.3f}, {p['ci95_high']:.3f}]" if p["ci95_low"] is not None else "requires three paired seeds"
            text.append(f"| {p['left']} − {p['right']} ({p['metric']}) | {p['difference']:.3f} | {interval} |")
        text += ["", "Student-t intervals use three paired seeds (df=2); exploratory and unadjusted across comparisons. Intervals including zero do not establish equivalence. Early stopping gives different training costs; this comparison measures the declared stopping protocol rather than equal successful-update budgets.",
                 "", "Euler K=1 is the main solver for every ODE here. Residual/conditioned/augmented are equivalent controls, not independent architecture discoveries. This study does not test whether deeper integration helps. Native GRU/LSTM AMP state precision differs from FP32 ODE accumulation. Matching parameter count changes some state or field widths.", ""]
        text += ["| Model | Exact parameters | State width | Latent width | Field width |", "|---|---:|---:|---:|---:|"]
        for model in protocol["models"]:
            entry = next(e for e in protocol["entries"] if e["task"] == task and e["model"] == model)
            values = entry["config_values"]
            text.append(f"| {model} | {entry['parameters']} | {values['hidden_dim']} | {values['latent_dim']} | {values['field_width']} |")
        text += ["", "State/latent/field configuration columns are architecture-specific; unused settings are retained in the configuration. Transformer uses its embedding width as model width.", "",
                 "Dynamics probes use FP32 selected examples. Field variation at fixed states measures dependence on tokens, and does not alone demonstrate useful learned selection. Finite sampled trajectories do not prove global stability."]
        for diagnostic in (d for d in diagnostics if d["task"] == task and d["seed"] == protocol["seeds"][0]):
            if "token_field_pairwise_l2_mean" in diagnostic:
                text.append(f"{diagnostic['model']} (seed {diagnostic['seed']}): mean token-field pairwise L2 {diagnostic['token_field_pairwise_l2_mean']:.4f}; cosine mean {diagnostic['token_field_cosine_mean']:.4f}; sampled trajectory finite: {diagnostic['finite_trajectory']}.")
            if "discretely_stable_fraction" in diagnostic:
                text.append(f"Structured decay: negative-A fraction {diagnostic['negative_fraction']:.3f}, discrete stable fraction {diagnostic['discretely_stable_fraction']:.3f}, maximum discrete amplification {diagnostic['max_discrete_amplification']:.4f}, median time constant {diagnostic['time_constant_median']:.4f} internal units.")
        # Descriptive, evidence-bound conclusions; never infer unobserved performance.
        if aggregates:
            metric = "test_bpc_mean" if task == "shakespeare" else "test_accuracy_mean"
            winner = min(aggregates, key=lambda a: a[metric]) if task == "shakespeare" else max(aggregates, key=lambda a: a[metric])
            text += [f"Among the currently tested conditions, {winner['model']} has the best mean {'test BPC' if task == 'shakespeare' else 'answer accuracy'}. This ranking is descriptive; use paired intervals to assess differences."]
            for model in ("conditioned", "autonomous", "structured", "gru", "transformer"):
                a = next((a for a in aggregates if a["model"] == model), None)
                if not a:
                    continue
                if task != "shakespeare":
                    text += [f"{model}: {'exceeds' if a['test_accuracy_mean'] > freq else 'does not exceed'} the frequency baseline in mean; memory-cue change causes {100*a['memory_control_drop_mean']:.2f} pp accuracy drop." if a["memory_control_drop_mean"] is not None else f"{model}: memory-control result unavailable."]
                else:
                    text += [f"{model}: {'improves on' if a['test_bpc_mean'] < bigram else 'does not improve on'} the bigram baseline in mean test BPC."]
        if len(tested) != expected:
            text += ["", "Study incomplete: remaining observations are listed in per_seed.csv; final cross-model conclusions are pending."]
        temporary = folder/"report.md.tmp"
        temporary.write_text("\n".join(text)+"\n")
        temporary.replace(folder/"report.md")
        index.append(f"- [{task.replace('_', ' ').title()}]({task}/report.md): {len(tested)}/{expected} finite independent-test results.")
        if plots and tested:
            task_lengths = [p for p in lengths if p["task"] == task and p.get("finite")]
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for a in aggregates:
                model = a["model"]
                ps = [p for p in task_lengths if p["model"] == model]
                xs = sorted({p["seq_len"] for p in ps})
                metric = "bpc" if task == "shakespeare" else "accuracy"
                scale = 1 if task == "shakespeare" else 100
                means = np.array([scale*mean_sd([p[metric] for p in ps if p["seq_len"] == x])[0] for x in xs])
                sds = np.array([scale*mean_sd([p[metric] for p in ps if p["seq_len"] == x])[1] for x in xs])
                axes[0].plot(xs, means, marker="o", label=model)
                axes[0].fill_between(xs, means-sds, means+sds, alpha=0.1)
                axes[1].scatter(a["training_seconds_mean"], a["test_bpc_mean"] if task == "shakespeare" else 100*a["test_accuracy_mean"], label=model)
            axes[0].set_xscale("log", base=2)
            axes[0].set(xlabel="Held-out window length", ylabel="Test BPC" if task == "shakespeare" else "Answer accuracy (%)")
            axes[1].set(xlabel="Measured training seconds (validation excluded)", ylabel="Test BPC" if task == "shakespeare" else "Answer accuracy (%)")
            axes[1].legend(fontsize=7)
            save(fig, folder/"length_and_compute.png")
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for model, runs in groups.items():
                train, val = defaultdict(list), defaultdict(list)
                for run in runs:
                    for p in read_csv(Path(run["run"])/"metrics.csv"):
                        train[int(p["step"])].append(float(p["train_loss"]))
                        if p["val_loss"]:
                            val[int(p["step"])].append(float(p["val_loss"]))
                    dynamics_plots(Path(run["run"]))
                xs = sorted(train)
                ys = [np.mean(train[x]) for x in xs]
                window = min(100, len(xs))
                axes[0].plot(xs[window-1:], np.convolve(ys, np.ones(window)/window, mode="valid"), label=model)
                xs = sorted(val)
                axes[1].plot(xs, [np.mean(val[x]) for x in xs], label=model)
            axes[0].set(xlabel="Attempted update", ylabel="Training CE, moving average")
            axes[1].set(xlabel="Attempted update", ylabel="Validation CE (surviving runs at each step)")
            axes[1].legend(fontsize=7)
            save(fig, folder/"learning_curves.png")
    temporary = output/"report.md.tmp"
    temporary.write_text("\n".join(index)+"\n")
    temporary.replace(output/"report.md")
    return rows
