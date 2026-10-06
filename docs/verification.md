# Implementation verification — 6 October 2026

Executed on the supplied RTX 2080 Ti machine, using Python 3.11 and PyTorch 2.7.0+cu126. The original system Python had no Torch; the existing isaaclab environment was used without modifying its packages.

## Checks completed

- **55 tests passed** in approximately four seconds. Tests cover all fixed solvers, convergence order, differentiability, every architecture, causality, masked task correctness, source/checkpoint handling, accumulation, and exact dropout-aware resume.
- Conditioned/explicit augmented trajectories agree within 1e-6 in float32 for Euler/Heun/RK4 and K=1/2/4/8; augmented x is exactly unchanged at every recorded boundary. Input/state gradients agree.
- All nine architectures overfit 32 fixed, length-32 associative-recall examples to loss <0.05 and 100% answer accuracy.
- Four-GPU DDP/FP16 training passed, including the structured negative control with an unused embedding and a projected-conditioned Heun/K=2 run.
- Six CUDA AMP tests verify FP32 ODE-state accumulation; projected models were rerun after enforcing the same solver-state precision as directly conditioned models. GRU/LSTM retain native Torch AMP cell precision.
- Tiny Shakespeare downloaded, trained for a smoke run, and evaluated at longer context lengths.
- Best/final checkpoint writing, source snapshot capture, evaluation, solver refinement, benchmarking, dynamics diagnostics, CSV aggregation, and plot generation executed successfully.

## Overfit gate

| Architecture | Updates/checks | Loss | Accuracy |
|---|---:|---:|---:|
| gru | 59 | 0.0462 | 100% |
| lstm | 92 | 0.0483 | 100% |
| residual | 53 | 0.0316 | 100% |
| autonomous | 94 | 0.0105 | 100% |
| conditioned | 53 | 0.0316 | 100% |
| augmented | 53 | 0.0316 | 100% |
| projected_conditioned | 63 | 0.0346 | 100% |
| structured | 48 | 0.0449 | 100% |
| transformer | 32 | 0.0342 | 100% |

The nine gates took 21.4 seconds of measured model runtime in the final run, plus interpreter startup. They use FP32, learning rate 0.003, weight decay zero, parameter seed 123 and fixed batch seed 7001. They test memorization, not held-out task performance. The two projected-model gates were rerun after the AMP precision correction and retained the same losses and step counts.

## First synthetic comparison

This is a deliberately small **one-seed engineering pilot**: 120 attempted AdamW updates, length 32, embedding dimension 16, eight value symbols, four key/value pairs, 16 examples/microbatch, and a matched target of 5,780 parameters. Best checkpoints are selected by validation cross entropy. Each validation has 64 answer symbols. Evaluation lengths are 32, 64, and 128. Full research plans separately retain the requested larger scales, three seeds and 2048-token evaluation.

| Model | Parameters | Best validation CE | Answer accuracy | AMP skipped updates |
|---|---:|---:|---:|---:|
| augmented | 5,756 | 2.0182 | 25.0% | 3 |
| autonomous | 5,836 | 2.0315 | 18.8% | 4 |
| conditioned | 5,756 | 2.0182 | 25.0% | 3 |
| gru | 5,780 | 2.1005 | 10.9% | 0 |
| lstm | 5,740 | 2.1062 | 10.9% | 0 |
| projected_conditioned | 5,720 | 2.0892 | 15.6% | 1 |
| residual | 5,756 | 2.0182 | 25.0% | 3 |
| structured | 5,794 | 2.0775 | 21.9% | 1 |
| transformer | 5,768 | 2.0577 | 17.2% | 1 |

All capacities are within 1.6% of the reference. Embeddings share dimensions and initial weights; they are independently trained. Models see identical training/validation streams. The conditioned, residual, and augmented models have identical final checkpoint weights (maximum absolute difference 0.0), identical best CE, and identical answer accuracy under Euler/K=1.

Chance answer accuracy is 12.5%. The largest observed pilot accuracy is 25%, but a single seed, 64 validation answers, early checkpoint selection, and limited optimization are insufficient to conclude superiority. Initial AMP overflow handling skipped 0–4 attempts depending on architecture; the CSV explicitly reports these events. This further limits interpreting tiny gaps. Recheck any substantive comparison with successful-update counts and numerical behavior. The code keeps the same AMP policy for every architecture.

The isolated single-device benchmarks in the pilot were rerun sequentially on physical GPU 2 after training completed. Actual values are in `results/verified_pilot/summary/summary.csv`; warmup is excluded. Earlier exploratory pilot outputs are superseded by `verified_pilot`.

## Resource probes

These are **untrained** models for planning, not quality results. The probes used FP16 AMP on otherwise idle physical GPU 1, embedding 128, state 256, sequence length 256. Training timing includes forward/backward/clipping/AdamW with LR=0, but excludes data generation, training monitors, validation and DDP communication. MiB below refers to PyTorch peak allocated memory, not total process VRAM.

| Model / solver | Batch | NFE/token | Parameters | Training tokens/s | Inference tokens/s | Peak train MiB |
|---|---:|---:|---:|---:|---:|---:|
| GRU | 8 | 0 | 321,473 | 8,420 | 32,194 | 83.2 |
| NODE Euler K=1 | 8 | 1 | 255,425 | 4,554 | 13,483 | 77.6 |
| NODE Heun K=4 | 2 | 8 | 255,425 | 200 | 615 | 82.1 |
| NODE RK4 K=8 | 16 | 32 | 244,260 | 394 | 1,194 | 494.9 |

GRU/Euler/Heun probes use Shakespeare vocabulary (65), while the RK4 K=8 probe uses the synthetic vocabulary (36), accounting for its smaller readout/parameter count. The RK4 probe demonstrates that the requested high-NFE setting fits comfortably in 11 GiB. Different microbatch sizes affect launch efficiency; do not compare the Heun batch-2 throughput with batch-8/16 numbers as a pure solver effect.

Compute-only estimates: the language default of 5,000 updates × 32 per-rank sequences × 256 tokens is about 1.35 hours for GRU and 2.5 hours for Euler K=1 at the measured rates. A synthetic RK4 K=8 run with 2,000 updates, batch 16 and length 256 is about 5.8 hours. Actual instrumented training, validation and DDP add overhead. DDP uses the same per-rank workload and processes four times the global examples; divide wall time by four only if global work is held fixed and measured scaling supports it.

The saved full plans contain 1,044 synthetic and 294 language configurations. Executing the reference-scale plans can take weeks to months. Compact configs reduce dimensions/budgets while preserving the scientific axes and three seeds; inspect convergence before interpreting negative results.

## Hypothesis status

| Question | Supported conclusion now |
|---|---|
| A: dynamics vs initial-condition conditioning | Unresolved; the one-seed pilot is insufficient. |
| B: benefit from K > 1 | Unresolved; higher-K correctness/resources were checked, but trained multi-K comparisons remain to run. |
| C: nonlinear vs structured | Unresolved; the pilot gap is too small and noisy to interpret. |
| D: best length extrapolation | Unresolved; short pilot curves are available, full length/seed comparisons are planned. |
| E: benefit justifies compute | Unresolved; throughput costs are measured, but reliable quality gains are not established. |
| F: useful token-selected vector fields | Token-conditioned diagnostics are available; field differences alone do not prove useful selection. Compare initial/learned fields and task behavior. |
| G: stable learned memory | Diagnostics and discrete-stability checks work. Negative A guarantees continuous diagonal decay, not numerical or projected recurrence stability. |
| H: beyond a residual recurrent cell | For Euler: no extra transition; exact equivalence is proved and the trained weights agree. For higher-order solvers: unresolved. |

## Artifacts

- `results/final_tests.log`: final test output.
- `results/overfit.json`: complete overfit configs and observations.
- `results/verified_pilot/summary/summary.csv`: nine-model comparison and compute measurements.
- `results/verified_pilot/summary/`: quality/compute and length plots.
- `results/language_smoke/plots/`: executed character-LM curves and dynamics plots.
- `results/structured_smoke/dynamics/`: structured coefficients and discrete stability diagnostics.
- `results/final_smoke/source_snapshot.zip`: source replay artifact.
- `results/synthetic_full/plan.json` and `results/language_full/plan.json`: planned configurations; full training was not launched.

Generated result artifacts are ignored by Git to keep checkpoint/data binaries out of source control. They remain present in the workspace. Reproduce them using the README commands.


## Completed three-seed follow-up

The requested focused follow-up is complete: 18 matched-capacity runs, three seeds, 1,000 updates each, independent synthetic testing through length 2048, query-key controls and isolated benchmarking. See [focused_comparison.md](focused_comparison.md) for the conclusions and `results/focused_recall/summary/focused_report.md` for full results. This adds evidence under one compact recall protocol; the broad research sweeps remain unexecuted. The updated suite passes 57 tests.
