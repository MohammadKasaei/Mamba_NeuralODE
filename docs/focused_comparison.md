# Focused three-seed experiment: conclusions

Completed 18 matched-capacity runs: four architectures across seeds 0/1/2, plus conditioned Euler K=4 and Heun K=4 across those seeds. All runs used 1,000 attempted updates, batch 64, length 64, embedding 32, reference memory/latent 64, eight pairs and sixteen values. The 22,308-parameter reference was matched within 0.1%.

Checkpoint selection used fixed validation CE. Each selected checkpoint was tested on an independent synthetic stream with 512 examples per seed at each of six lengths, including 2048. All 18 runs completed; all test trajectories were finite. Isolated benchmarks ran sequentially on the same GPU and had zero skipped updates. Training runs used 0–5 initial AMP skips, recorded in the CSV.

| Model / solver | Independent length-64 accuracy, mean ± seed SD | Length-2048 accuracy | Training benchmark tokens/s |
|---|---:|---:|---:|
| gru | 15.10% ± 0.30 pp | 10.29% | 63,855 |
| autonomous | 8.14% ± 1.71 pp | 9.90% | 28,975 |
| structured | 16.41% ± 1.03 pp | 6.64% | 44,441 |
| conditioned_euler_k1 | 24.74% ± 0.60 pp | 5.86% | 36,866 |
| conditioned_euler_k4 | 24.15% ± 0.79 pp | 6.71% | 12,287 |
| conditioned_heun_k4 | 24.41% ± 0.52 pp | 5.66% | 6,306 |

**The most frequent context-value baseline scores 26.24%, without reading the query key.** Uniform guessing is 6.25%; the lookup oracle scores 100%. The trained conditioned model’s mean accuracy is below the frequency baseline, and changing its query to another present key produces exactly zero mean accuracy drop (sample SD 0.34 percentage points). This is evidence that reliable key-specific retrieval has not been learned, despite the apparent advantage over the other trained models. The below-baseline point estimate is not a claim of statistically established inferiority to that baseline.

## What this supports

- Dynamics conditioning beats initial-condition-only conditioning on measured accuracy under this matched-capacity, compact, fixed-update protocol. The paired difference is +16.60 pp, with exploratory 95% interval [12.94, 20.26]. This does not establish the selective-memory mechanism in H1: the task-specific controls expose a frequency shortcut.
- Increasing solver depth gives no demonstrated benefit: Euler K=4 changes accuracy by −0.59 pp [−2.81, 1.64], and Heun K=4 by −0.33 pp [−1.89, 1.23], relative to Euler K=1. Their isolated training throughput is approximately 3.0× and 5.8× lower. These intervals include zero and do not prove equivalence or rule out small gains.
- The nonlinear conditioned model beats the implemented stable diagonal structured model at this budget (+8.33 pp [6.85, 9.82]), but neither solves recall. This does not establish a general advantage over SSMs, other structured parameterizations, or other tasks.
- Conditioned models fall to roughly chance accuracy at length 2048, so H4 is unsupported in this experiment. GRU and autonomous models retain about 10%, still far below the frequency-only baseline.
- For Euler, the residual-cell equivalence remains exact. These results provide no additional evidence that the continuous interpretation improves task performance or compute efficiency.

## Limits and next useful controls

Only one synthetic task, one training length, compact dimensions and a fixed 1,000-update budget were tested. The scheduler ends at zero LR, so a flat final curve is not proof of convergence. Three seeds give limited statistical power; paired intervals are exploratory and unadjusted for multiple comparisons.

Parameter matching does not equalize memory paths: the autonomous model learns P/Q projections while the conditioned model carries memory directly, and the structured model has a larger state. Native GRU AMP precision differs from FP32 ODE accumulation. These are limits on a causal claim about input entry alone.

A useful next experiment would train on query-balanced contexts with distinct values, making frequency shortcuts weaker, and check query-key sensitivity before spending more compute on solver depth. Replicate at larger capacity and longer optimization budgets before generalizing these findings to NODEs or language modeling.

## Reproducibility

Run `python scripts/run_focused_comparison.py --execute --gpus 0 1 2 3` after the README setup. Existing completed runs are reused; use a new output directory to change the protocol. Every run saves resolved configuration, seed, source hashes/snapshot, optimizer/scaler/RNG checkpoints and step metrics.

Primary results are in `results/focused_recall/summary/test_summary.csv`; full numerical analysis is in `focused_report.md`, `seed_aggregates.csv`, `paired_contrasts.json` and `shortcut_baselines.csv`. Figures include independent accuracy, query-blind baselines, length extrapolation, learning curves and quality versus compute.
