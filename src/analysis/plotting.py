"""Aggregate saved artifacts; never invent observations for unexecuted runs."""
import csv
import hashlib
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as file:
        return list(csv.DictReader(file))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dynamics_plots(run):
    output = run/"plots"
    output.mkdir(exist_ok=True)
    rows = read_csv(run/"dynamics/token_dynamics.csv")
    if rows:
        fig, ax = plt.subplots()
        for example in sorted({r["example"] for r in rows}):
            subset = [r for r in rows if r["example"] == example]
            ax.plot([int(r["position"]) for r in subset], [float(r["hidden_norm"]) for r in subset], label=f"example {example}")
        ax.set(xlabel="Token position", ylabel="Memory norm")
        ax.legend()
        save(fig, output/"hidden_norm_tokens.png")
    rows = read_csv(run/"dynamics/internal_trajectory.csv")
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        subset = [r for r in rows if r["example"] == "0"]
        positions = sorted({int(r["position"]) for r in subset})
        for pos in positions[:8]:
            part = [r for r in subset if int(r["position"]) == pos]
            for ax, metric in zip(axes, ("hidden_norm", "derivative_norm")):
                ax.plot([float(r["tau"]) for r in part], [float(r[metric]) for r in part], label=f"token {pos}")
                ax.set(xlabel="Internal pseudo-time", ylabel=metric)
        axes[0].legend(fontsize=7)
        save(fig, output/"internal_trajectory.png")
    states_path = run/"dynamics/internal_states.pt"
    if states_path.exists():
        import torch
        states = torch.load(states_path, weights_only=True)[0, :8].numpy()
        flat = states.reshape(-1, states.shape[-1])
        centered = flat-flat.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected = (centered @ vt[:2].T).reshape(*states.shape[:2], 2)
        fig, ax = plt.subplots()
        for pos, coordinates in enumerate(projected):
            ax.plot(coordinates[:, 0], coordinates[:, 1], marker=".", label=f"token {pos}")
            ax.scatter(*coordinates[0], marker="x")
        ax.set(xlabel="PC 1", ylabel="PC 2", title="Internal trajectories; x marks tau=0")
        ax.legend(fontsize=7)
        save(fig, output/"internal_state_pca.png")
    rows = read_csv(run/"dynamics/structured_coefficients.csv")
    if rows:
        values = [float(r["timescale"]) for r in rows if r["timescale"] and float(r["timescale"]) > 0]
        if values:
            fig, ax = plt.subplots()
            ax.hist(np.log10(values), bins=40)
            ax.set(xlabel="log10 time constant (-1/A), internal time units", ylabel="Channel-token count")
            save(fig, output/"timescales.png")
    cosine_path = run/"dynamics/token_field_cosines.pt"
    if cosine_path.exists():
        import torch
        values = torch.load(cosine_path, weights_only=True).numpy()
        fig, axes = plt.subplots(1, len(values), figsize=(10, 4))
        for i, ax in enumerate(np.atleast_1d(axes)):
            im = ax.imshow(values[i], vmin=-1, vmax=1, cmap="coolwarm")
            ax.set(xlabel="Token id", ylabel="Token id", title=("Zero state" if i == 0 else "Random state"))
            fig.colorbar(im, ax=ax)
        save(fig, output/"token_field_cosines.png")


def aggregate(root="results", output=None):
    root = Path(root)
    output = Path(output) if output else root/"summary"
    output.mkdir(parents=True, exist_ok=True)
    summary, extrapolations, curves, failures = [], [], [], []
    for path in sorted(root.rglob("metadata.json")):
        run = path.parent
        meta = json.loads(path.read_text())
        import yaml
        cfg = yaml.safe_load((run/"config.yaml").read_text())
        failure_path = run/"failure.json"
        if failure_path.exists():
            failure = json.loads(failure_path.read_text())
            failures.append(dict(run=str(run), status="failed", model=cfg["model"], task=cfg["task"], seed=cfg["seed"], integrator=cfg["integrator"], K=cfg["steps_per_token"], parameters=meta["parameters"], train_length=cfg["seq_len"], error_type=failure["error_type"], error=failure["message"]))
            continue
        metrics = read_csv(run/"metrics.csv")
        if not metrics:
            continue
        validation_rows = [r for r in metrics if r.get("val_loss")]
        if not validation_rows:
            continue
        best = min(validation_rows, key=lambda r: float(r["val_loss"]))
        bench = json.loads((run/"benchmark.json").read_text()) if (run/"benchmark.json").exists() else {}
        comparison_cfg = {k: v for k, v in cfg.items() if k not in ("seed", "output_dir", "resume", "device", "diagnostics_examples", "diagnostics_tokens")}
        comparison_id = hashlib.sha256(json.dumps(comparison_cfg, sort_keys=True).encode()).hexdigest()[:12]
        row = dict(run=str(run), status="completed" if int(metrics[-1]["step"]) >= cfg["train_steps"] else "partial", comparison_id=comparison_id, device=meta.get("device"), model=cfg["model"], task=cfg["task"], seed=cfg["seed"],
                   integrator=cfg["integrator"], K=cfg["steps_per_token"], embedding_dim=cfg["embedding_dim"],
                   hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], field_width=cfg["field_width"],
                   train_length=cfg["seq_len"], train_steps=int(metrics[-1]["step"]), best_step=int(best["step"]),
                   parameters=meta["parameters"], world_size=meta["world_size"], effective_batch=meta["effective_batch"],
                   amp=cfg["amp"], match_parameters=cfg["match_parameters"],
                   damping=cfg["damping"], gated=cfg["gated"], stable_a=cfg["stable_a"],
                   a_conditioned=cfg["a_conditioned"], b_conditioned=cfg["b_conditioned"], structured_drive=cfg["structured_drive"],
                   val_loss=float(best["val_loss"]), val_accuracy=float(best["val_accuracy"]), val_bpc=float(best["val_bpc"]),
                   training_seconds=float(metrics[-1]["training_seconds"]),
                   cost_to_best_seconds=float(best["training_seconds"]),
                   nfe_per_token=meta["nfe_per_token"],
                   train_tokens_per_sec=bench.get("train_tokens_per_sec", float(np.median([float(r["train_tokens_per_sec"]) for r in metrics]))),
                   inference_tokens_per_sec=bench.get("inference_tokens_per_sec", ""),
                   peak_memory_bytes=max(int(r["peak_memory_bytes"]) for r in metrics),
                   forward_flops_per_token_estimate=bench.get("forward_flops_per_token_estimate", ""),
                   nan_inf_events=max(int(r["nan_inf_events"]) for r in metrics),
                   git_commit=meta.get("git_commit"), git_dirty=meta.get("git_dirty"))
        summary.append(row)
        curves.append((row, metrics))
        extrap_path = run/"extrapolation.json"
        if extrap_path.exists():
            extrapolations.append((row, json.loads(extrap_path.read_text())["metrics"]))
        dynamics_plots(run)
        if cfg["task"] == "shakespeare":
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot([int(r["step"]) for r in metrics], [float(r["train_loss"]) for r in metrics], label="train")
            val = [r for r in metrics if r["val_loss"]]
            axes[0].plot([int(r["step"]) for r in val], [float(r["val_loss"]) for r in val], label="validation")
            axes[0].set(xlabel="Training step", ylabel="Cross entropy (nats)")
            axes[0].legend()
            axes[1].plot([int(r["step"]) for r in val], [float(r["val_bpc"]) for r in val])
            axes[1].set(xlabel="Training step", ylabel="Validation BPC")
            save(fig, run/"plots/language_learning_curve.png")
    if not summary and not failures:
        raise ValueError(f"No completed validations or failures under {root}")
    with (output/"summary.csv").open("w", newline="") as file:
        all_rows = summary+failures
        columns = list(dict.fromkeys(k for row in all_rows for k in row))
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    # Comparisons are separated by task and training length; each point is a run.
    for task in sorted({r["task"] for r in summary}):
        lengths = sorted({r["train_length"] for r in summary if r["task"] == task})
        for length in lengths:
            rows = [r for r in summary if r["task"] == task and r["train_length"] == length]
            folder = output/f"{task}_L{length}"
            folder.mkdir(exist_ok=True)
            quality_key = "val_bpc" if task == "shakespeare" else "val_accuracy"
            for x_key, name, xlabel in (("K", "quality_vs_K", "Integration steps K"),
                        ("training_seconds", "quality_vs_wallclock", "Training seconds (validation excluded)"),
                        ("cost_to_best_seconds", "quality_vs_cost_to_best", "Training seconds to best checkpoint"),
                        ("parameters", "quality_vs_parameters", "Trainable parameters"),
                        ("nfe_per_token", "quality_vs_nfe", "Forward field evaluations / token"),
                        ("train_tokens_per_sec", "quality_vs_throughput", "Training tokens / second")):
                fig, ax = plt.subplots()
                for model in sorted({r["model"] for r in rows}):
                    part = [r for r in rows if r["model"] == model]
                    ax.scatter([r[x_key] for r in part], [r[quality_key] for r in part], label=model, alpha=0.7)
                ax.set(xlabel=xlabel, ylabel=quality_key)
                ax.legend(fontsize=7)
                save(fig, folder/f"{name}.png")
            fig, ax = plt.subplots()
            for model in sorted({r["model"] for r in rows}):
                part = [r for r in rows if r["model"] == model]
                ax.scatter([r["K"] for r in part], [r["train_tokens_per_sec"] for r in part], label=model)
            ax.set(xlabel="K", ylabel="Training tokens / second")
            ax.legend(fontsize=7)
            save(fig, folder/"throughput_vs_K.png")
            for metric, name, ylabel, scale in (("parameters", "parameter_comparison", "Trainable parameters", 1),
                    ("peak_memory_bytes", "peak_memory_comparison", "Peak training allocated memory (MiB)", 2**20)):
                fig, ax = plt.subplots(figsize=(10, 4))
                labels = [f"{r['model']}\nK={r['K']},s={r['seed']}" for r in rows]
                ax.bar(range(len(rows)), [r[metric]/scale for r in rows])
                ax.set_xticks(range(len(rows)), labels, rotation=90, fontsize=6)
                ax.set(ylabel=ylabel)
                save(fig, folder/f"{name}.png")
            relevant = [(r, e) for r, e in extrapolations if r["task"] == task and r["train_length"] == length]
            if relevant:
                fig, ax = plt.subplots()
                # Mean/std across seeds only for identical non-seed configuration.
                grouped = defaultdict(list)
                for r, e in relevant:
                    key = tuple((k, v) for k, v in r.items() if k in ("model", "integrator", "K", "comparison_id", "world_size"))
                    grouped[key].append(e)
                for key, entries in grouped.items():
                    points = defaultdict(list)
                    for entry in entries:
                        for e in entry:
                            if e.get("finite", True) is False:
                                continue
                            points[e["seq_len"]].append(e["bpc" if task == "shakespeare" else "accuracy"])
                    xs = sorted(points)
                    if not xs:
                        continue
                    means = np.array([np.mean(points[x]) for x in xs])
                    std = np.array([np.std(points[x], ddof=1) if len(points[x]) > 1 else 0 for x in xs])
                    config = dict(key)
                    ax.plot(xs, means, marker="o", label=f"{config['model']} {config['integrator']} K={config['K']} n={len(entries)}")
                    ax.fill_between(xs, means-std, means+std, alpha=0.15)
                ax.set(xlabel="Evaluation length", ylabel="Validation BPC" if task == "shakespeare" else "Answer accuracy")
                ax.set_xscale("log", base=2)
                ax.legend(fontsize=6)
                save(fig, folder/"quality_vs_sequence_length.png")
    text = ["# Evidence status", "", f"{len(summary)} runs with validation results; {len(failures)} recorded failures (also in summary.csv).",
            "Plots display observed runs; differing capacity, solver, or ablations must be controlled before causal comparison.",
            "", "The overfit gate verifies implementation, not generalization. Synthetic BPC/perplexity are over scored answer tokens, not language metrics.",
            "", "For A–H, compare paired seeds at equal training budget and capacity and report mean differences and uncertainty:",
            "A. autonomous vs conditioned vs projected_conditioned.",
            "B. K=1 vs K=2/4/8 within the same model and solver, plus evaluation-time solver refinement.",
            "C. conditioned vs structured at matched parameters and fixed state size.",
            "D. compare held-out length curves, including collapse at the longest lengths.",
            "E. compare quality against time, throughput, parameters, and NFE; training FLOPs differ from NFE.",
            "F. inspect token_field_cosines and zero-field degeneracy.",
            "G. inspect finite trajectories, memory norms and A/exp(A); negative A alone does not prove discretization stability.",
            "H. Euler NODE is mathematically identical to the residual control. Any claimed benefit must survive that control and compute accounting.", ""]
    seed_counts = defaultdict(set)
    for r in summary:
        seed_counts[(r["task"], r["model"])].add(r["seed"])
    text += ["Independent seeds per task/model:"]+[f"- {task}/{model}: {len(seeds)}" for (task, model), seeds in sorted(seed_counts.items())]
    if any(len(s) < 3 for s in seed_counts.values()):
        text += ["", "At least one comparison has fewer than three seeds. Hypotheses remain unresolved; do not interpret pilot results as research conclusions."]
    (output/"evidence.md").write_text("\n".join(text))
    return summary+failures
