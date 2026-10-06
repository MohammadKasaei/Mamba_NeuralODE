# All-task training protocol

The new study runs nine architectures on selective copying, associative recall,
induction and Tiny Shakespeare, with seeds 0, 1 and 2: 108 independent training
runs. Previous recall follow-up and language smoke outputs are separate. The
initial small-batch queue in `results/all_tasks` was stopped at the user's request
to increase batch size. Its checkpoints/logs remain preserved; the active study
uses `results/all_tasks_large_batch` and restarts every model from the same seeds.

This is the main architecture study, using Euler K=1. It does not execute the
1,338-configuration solver/state/conditioning ablation matrix. The residual,
conditioned and augmented Euler controls are deliberately all trained and
reported, although they implement equivalent transitions.

Reference embedding/state/latent widths are 64/128/128, versus 32/64/64 in the
previous recall follow-up and 128/256/256 in the original reference-scale plan.
Each task matches all models within 1% of its GRU parameter count; the resolved
state and field widths are saved individually. This explicit scale reduction
keeps the full main architecture comparison feasible on four RTX 2080 Ti cards.

Synthetic training uses length 64 and batch **1,024**; language training uses length
256 and batch **512**. Each GPU executes an independent job with the same global
batch within a task. AdamW uses LR 0.001, weight decay 0.01, clipping 1.0,
100-step warmup and cosine decay over a ceiling of 20,000 attempted updates.
Native GRU/LSTM AMP state precision differs from FP32 ODE state accumulation.

Validation runs every 25 attempts. After a minimum of 100 updates (the end of
warmup), stopping occurs after ten validation checks with no cumulative
CE improvement of at least 0.001 relative to the last significant improvement.
The checkpoint with the lowest actual validation CE is always saved, including
improvements smaller than the stopping threshold. Optimizer, scaler, RNG and
stopping state are checkpointed for exact resume. A plateau under this optimizer
and stopping criterion does not establish global convergence. The minimum now
processes 102,400 synthetic examples or 51,200 language windows, exceeding the
previous 32,000-example minimum in each task. Retaining the old minimum update
counts after a 32-fold batch increase would force unnecessary training before
stopping could take effect. The 20,000-update ceiling, LR and patience/CE threshold
are unchanged; different update/data budgets are explicit in each saved protocol.

Synthetic training and validation use disjoint random streams. Final testing
adds a seed offset of 10^12 and uses 512 fresh examples per seed and length.
Evaluation batch size is separate from training: 128 for synthetic tasks and 64
for language, with respectively four and two batches. The validation/test example
counts are unchanged by the training batch increase. Long-context test batches
allow up to 65,536 tokens, reducing small-batch overhead while preserving counts.
Copying scores four answer tokens per example and also reports whole-sequence
accuracy; preceding answers are teacher-forced. The number of pairs/copied
symbols remains fixed during length extrapolation.

Shakespeare has contiguous 80% training, 10% checkpoint-selection validation and
10% reserved test text. Its final test uses 128 sampled windows per seed and
length. Windows may overlap within the same partition and are not statistically
independent character samples. No window crosses a partition boundary. Test
text does not select checkpoints or determine stopping. Long contexts start with
fresh state and score all positions; this is window-length quality evaluation,
not the synthetic retrieval-distance challenge.

All tasks are tested through length 2048. Long-window evaluation reduces the
microbatch while preserving the example count. Shortcut baselines use precisely
the same seeded batches. Synthetic reports include frequency, recent-symbol and
lookup-oracle controls; language reports include training-text unigram and
add-one-smoothed bigram baselines. Cue-removal/query-change controls preserve
original labels and measure input sensitivity; their modified input distribution
limits causal interpretation.

Before launch, run the test suite and all nine architecture overfit gates for
each synthetic task. Untrained resource probes estimate runtime only; they are
not quality results. Training/testing jobs share a four-GPU queue. Final compute
benchmarks run sequentially on one common GPU after the queue finishes, excluding
validation, data generation and checkpoint IO from their timing. Training CSV
times include data generation and device transfer.

Generate a plan, then execute or resume the identical plan:

```bash
OMP_NUM_THREADS=2 /home/bayespc/miniconda3/envs/isaaclab/bin/python scripts/run_all_tasks.py
OMP_NUM_THREADS=2 /home/bayespc/miniconda3/envs/isaaclab/bin/python scripts/run_all_tasks.py --execute --gpus 0 1 2 3
```

`--execute --probes-only` runs preflight and resource probes without training.
`--tasks`, `--models`, and `--seeds` allow explicitly smaller subsets. Use a new
`--output` directory when changing a protocol; existing configurations cannot
silently change. A lock prevents duplicate runners in one output directory.

Monitor `results/all_tasks_large_batch/progress.json`, per-run `metrics.csv`, and
`results/all_tasks_large_batch/logs/`. A final `completion.json` is created only after
training, testing, isolated benchmarking and report generation finish.

The generated `results/all_tasks_large_batch/summary/report.md` links one detailed report per
task. Reports include seed means/sample SD, exact capacities, best/final losses,
stopping steps, AMP skips, quality/cost, length curves, memory controls and
dynamics. `all_runs.csv` includes every planned run, including pending or failed
ones. `length_results.csv` and `dynamics_summary.csv` preserve individual results.
Per-task folders contain baselines, per-seed tables, seed aggregates, paired
exploratory 95% t intervals, and learning/length/compute plots. Three-seed
intervals have limited power and are unadjusted for multiple comparisons.

Refresh reports during training without changing the protocol:

```bash
OMP_NUM_THREADS=2 /home/bayespc/miniconda3/envs/isaaclab/bin/python -c \
  'from src.analysis.all_tasks import report; report("results/all_tasks_large_batch", plots=False)'
```

Missing observations are always explicit. Consult `progress.json` and
`completion.json` to distinguish running work from a completed study.

## Batch-size calibration

The batch increase was measured on actual parameter-matched configurations with
the state-statistics kernels used by training. Values below are resident-batch
training tokens/s, excluding data generation and transfers. They use untrained
weights, AMP scale 64, three warmup and five timed iterations; all timed updates
succeeded. These resource probes are not task-quality results.

| Task / model | Previous batch | Tokens/s | New batch | Tokens/s | Ratio |
|---|---:|---:|---:|---:|---:|
| Synthetic GRU | 32 | 21,309 | 1,024 | 595,101 | 27.9× |
| Synthetic conditioned NODE | 32 | 13,835 | 1,024 | 491,096 | 35.5× |
| Shakespeare GRU | 16 | 13,633 | 512 | 463,050 | 34.0× |
| Shakespeare conditioned NODE | 16 | 9,082 | 512 | 303,292 | 33.4× |

The GRU, conditioned NODE, structured NODE and Transformer all fit the chosen
batches. Largest peak allocated training memory among these probes was 957 MiB
(structured Shakespeare). Other models are checked by the queue's resource
preflight before full training. Each model's batch variants ran on the same GPU;
different models calibrated concurrently on separate GPUs. The final trained
benchmarks run on one common GPU. Short calibration timings are estimates, not
guarantees of end-to-end speedup. At synthetic batch 1,024, CPU batch generation
also takes about 80 ms and becomes a significant part of each update.

Raw measurements are in `results/batch_calibration/*.json`. Reproduce a scan with:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python scripts/profile_batch_sizes.py \
  --config results/all_tasks_large_batch/configs/associative_recall_gru_s0.yaml \
  --batches 32 256 512 1024 --output results/batch_calibration/replay_gru.json
```
