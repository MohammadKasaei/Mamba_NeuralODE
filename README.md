# Token-conditioned Neural ODE language models

A reproducible PyTorch project testing whether a token should enter recurrent dynamics through an initial condition, a constant control, or both. Includes nine architectures, fixed-step numerical controls, three dynamic synthetic tasks, character-level Tiny Shakespeare, single-device/DDP FP16 training, diagnostics, benchmarks, and planned ablations. It is designed for 4 × RTX 2080 Ti (11 GiB each).

The scientific objective is an interpretable comparison, including negative results. **Euler conditioned NODE and the residual control are exactly the same model when their weights and K match.** Explicit augmentation is a correctness experiment, not additional model capacity.

## Installation

Use Python 3.11. The system's default Python may differ. In the development machine the existing `/home/bayespc/miniconda3/envs/isaaclab/bin/python` provided PyTorch 2.7.0+cu126; no packages in that environment were changed.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

The CUDA 12.6 wheel supports Turing/SM75; a compatible NVIDIA driver is required. For CPU debugging, install the same Torch release from the CPU wheel index. FP16 AMP is enabled on CUDA only. BF16 is never assumed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated plugins from the host environment.

## Mathematical implementation

All recurrent models start each window with zero memory. For LSTM both h and c start at zero. Inputs are token embeddings x_t = E[token_t]. Output is an untied affine readout `W_out h_t + b_out`, predicting the next token. There is no recurrence across independently sampled training windows. A caller may explicitly carry recurrent state across contiguous chunks via `forward(..., state=state, return_stats=True)`.

Define the generic vector field:

```text
u = LayerNorm(h)
q = W_h u + W_x x
F(h,x) = W_2 SiLU(W_1 q + b_1) + b_2
```

`W_h` and `W_x` have no biases. Optional gate multiplies this nonlinear part by `sigmoid(W_gate q + b_gate)`. Optional damping subtracts `softplus(lambda) * h`, with one learned lambda per state channel initialized to -2. The autonomous field G(s) uses the same formula without W_x x. Embedding dimension remains fixed across capacity matching.

| Config `model` | Exact token transition |
|---|---|
| `gru` | h_t = GRUCell(x_t, h_(t-1)) |
| `lstm` | (h_t,c_t) = LSTMCell(x_t,(h_(t-1),c_(t-1))) |
| `residual` | h^0=h_(t-1); h^(k+1)=h^k + F(h^k,x_t)/K; h_t=h^K |
| `autonomous` | s(0)=P([h_(t-1);x_t]); ds/dtau=G(s); h_t=Q(s(1)) |
| `conditioned` | h(0)=h_(t-1); dh/dtau=F(h,x_t); h_t=h(1) |
| `augmented` | z(0)=[h_(t-1);x_t]; dz/dtau=[F(h,x);0]; h_t=z_h(1) |
| `projected_conditioned` | s(0)=P([h_(t-1);x_t]); ds/dtau=F(s,x_t); h_t=Q(s(1)) |
| `structured` | h(0)=h_(t-1); dh/dtau=a(x_t) elementwise-multiply h + b(x_t) |
| `transformer` | Two causal pre-norm encoder layers by default, sinusoidal positions, GELU FFN, affine readout |

P and Q are learned affine maps, with separate `latent_dim` and `hidden_dim`. Each solve covers tau in [0,1], with x fixed. Parameters are shared across internal steps and sequence positions.

Structured coefficients are computed **once per token**:

```text
r(x) = a_bias + W_A x             (omit W_A x when a_conditioned=false)
a(x) = -softplus(r(x))            (or r(x) when stable_a=false)

structured_drive=bias:
    b(x) = b_bias + W_b x         (omit W_b x when b_conditioned=false)

structured_drive=matrix:
    b(x) = b_bias + B(x) x
    B(x) = diag(sigmoid(W_gate x + c_gate)) W_b   if b_conditioned=true
    B(x) = W_b                                  otherwise
```

The matrix variant uses a factored, input-gated projection rather than materializing an input-dependent N×E matrix from an E→NE network. In the matrix variant `b_conditioned=false` means **constant B**, not zero token influence. In the bias variant, making both A and b independent of x removes token access entirely and is a deliberate negative control. DDP handles the resulting unused embedding.

## Numerical controls

All solvers use dt=1/K with K in {1,2,4,8} in the generated sweeps:

```text
Euler: k1=f(y); y_next=y+dt*k1
Heun:  k1=f(y); k2=f(y+dt*k1); y_next=y+dt*(k1+k2)/2
RK4:   k1=f(y); k2=f(y+dt*k1/2); k3=f(y+dt*k2/2); k4=f(y+dt*k3)
       y_next=y+dt*(k1+2*k2+2*k3+k4)/6
```

Forward field evaluations/token are K, 2K, and 4K. Counts exclude backward recomputation and additional diagnostic probes. An SSM field evaluation is cheaper than an MLP field evaluation; NFE alone is not a fair compute metric. No adaptive solver is installed or used. Optional Dopri5 and larger tokenized datasets are deferred until the core experiments are interpreted.

## Repository tree

```text
configs/              smoke.yaml, synthetic.yaml, language.yaml, *_compact.yaml, ablations.yaml
src/
  config.py           strict dataclass/YAML configuration
  models/             base, factory, gru, lstm, transformer, vector_fields,
                      integrators, conditioned_node, projected_node,
                      augmented_node, structured_node
  data/               synthetic, tiny_shakespeare, factory
  training/           trainer, distributed, metrics
  analysis/           dynamics, benchmarking, plotting
scripts/              train, eval, overfit, download_data, benchmark, analyze,
                      refine_solver, sweeps, run_synthetic_sweep, run_language_sweep
tests/                integration, equivalence, shapes/causality, task correctness,
                      Shakespeare split, accumulation and exact resume
results/              generated checkpoints, metrics, plans and plots (git ignored)
docs/                 verification.md
requirements.txt      exact tested direct dependency versions
pyproject.toml         package/test configuration
```

## Smoke and correctness gates

Do these before a substantial run. Sweep execution runs the tests and the architecture overfit gate automatically.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 python -m pytest -q
python scripts/overfit.py --output results/overfit.json
python scripts/train.py --config configs/smoke.yaml --set output_dir=results/smoke_replay
python scripts/eval.py results/smoke_replay/best.pt --lengths 32 64 --batches 2
python scripts/benchmark.py results/smoke_replay/best.pt --repeats 3
python scripts/analyze.py --root results --output results/summary
```

Overfit uses 32 **fixed** sequences of length 32, FP32, AdamW, clipping, and a strict loss <0.05 gate. Every architecture must pass. It is an implementation test and provides no validation evidence. Change `--task` to `selective_copying` or `induction` for additional task-specific gates. Tests verify float32 augmented/conditioned trajectories at every boundary (`max_abs_error < 1e-6`), unchanged augmented x, input/state gradients, exact residual/Euler equality, numerical convergence orders, causality, state carry, partition-invariant accumulation, and exact checkpoint resume including dropout.

To debug on CPU append `--set device=cpu amp=false` to train, or `--device cpu` to eval/benchmark/overfit. Use a new output directory when rerunning; existing completed experiments are not overwritten.

## Synthetic experiments on one GPU

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/synthetic.yaml \
  --set model=conditioned task=associative_recall seed=0 \
  steps_per_token=4 integrator=heun output_dir=results/recall_node_heun4
python scripts/eval.py results/recall_node_heun4/best.pt
python scripts/benchmark.py results/recall_node_heun4/best.pt
python scripts/refine_solver.py results/recall_node_heun4/best.pt
```

Use the same `match_parameters` target for comparisons (the main sweep chooses the default GRU's exact count). Direct single runs default to fixed dimensions and report their actual capacity rather than claiming matching.

Synthetic training is generated dynamically; validation uses a fixed, disjoint RNG stream. Within each seed, architectures see the same effective batches, including when accumulation partitions change. Synthetic tasks score only masked answer tokens:

- **Selective copying:** marked symbols are dispersed among unmarked distractors; after COPY, predict the marked symbols in order. Four copied items by default. Previous output symbols are teacher-forced, as in next-token modeling.
- **Associative recall:** unique key/value pairs are dispersed in a padded stream; a final key plus QUERY asks for its value. Eight pairs by default.
- **Induction:** a random stream contains one unambiguous q,v bigram; its final repeated q asks for v. Keys and values share the same vocabulary, and the query key is excluded from other context positions.

Length extrapolation holds the number of copying items/recall pairs constant, increasing their delays and distraction. It tests memory distance rather than increasing dictionary size. The final answer is hidden at its prediction position. PAD is an ordinary input; there is no information-bearing attention padding mask.

## Four-GPU DDP

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/synthetic.yaml \
  --set model=conditioned seed=0 amp=true output_dir=results/ddp_recall

torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/smoke.yaml \
  --set amp=true output_dir=results/ddp_smoke
```

All GPUs hold a complete model. DDP distributes independent batches; recurrence remains sequential in token positions. `batch_size` is **per rank per microbatch**, so effective global batch is `batch_size × accumulation_steps × WORLD_SIZE`. Fix world size across comparisons. DDP validation all-reduces loss sums, correct answers, and supervised counts; checkpoint decisions are shared. Nonfinal accumulation microbatches use `no_sync`. LR follows a shared linear-warmup/cosine schedule over optimizer steps. Gradient norms are measured after unscaling and before clipping. Training-step budgets count attempted updates; FP16 overflow can skip updates, which are explicitly logged. Compare successful updates as well when skips are material. Linear operations use FP16 AMP with FP32 parameters and FP32 ODE-state accumulation; GRU/LSTM use their native Torch AMP cell precision; no BF16 or model parallelism.

The trainer logs loss, LR, gradient/memory/update norms, NaN/Inf events, AMP scale/skipped updates, synchronized training tokens/s, ms/token, cumulative training time, peak allocated memory, and validation quality. Fatal numerical failures save `failure.json`. Synthetic examples have the same target count; language examples have equal length, so accumulated and DDP loss normalization weights every supervised token equally.

## Tiny Shakespeare

```bash
python scripts/download_data.py
python scripts/train.py --config configs/language.yaml \
  --set model=gru output_dir=results/shakespeare_gru seed=0

torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/language.yaml \
  --set model=conditioned output_dir=results/shakespeare_node seed=0
python scripts/eval.py results/shakespeare_node/best.pt
python scripts/benchmark.py results/shakespeare_node/best.pt
```

The data source is Karpathy's public Tiny Shakespeare text. The loader saves SHA256 and vocabulary and uses a contiguous 90%/10% split. Training/validation windows never cross that split. Defaults: L=256, embedding=128, memory=256, per-rank batch=8, accumulation=4. Validation reports CE in nats, BPC=CE/log(2), perplexity=exp(CE). Synthetic BPC/perplexity concern answer symbols only and should not be presented as language metrics.

Long-context validation uses fresh zero memory or full Transformer context for each sampled validation window; it scores all next-character positions. It measures average window quality as reset overhead decreases, not a strict retrieval-length challenge. Transformers use sinusoidal positions and can run at 2048, but have quadratic attention cost. Inference benchmarks use teacher-forced full windows, not cached autoregressive generation.

## Reproducible sweeps

Planning is the default. `--execute` starts sequential experiments, with one or four GPUs per experiment. Each planned run gets its own YAML and directory. At least three seeds are the research default. The `pilot` suite deliberately uses one seed, length 32, small widths, and 120 updates; it is an engineering check only.

```bash
# First short comparison on one GPU, after gates:
python scripts/run_synthetic_sweep.py --suite pilot --output results/pilot_replay --execute

# Parameter-matched main models, all three tasks, seeds 0/1/2, lengths 64/128/256:
python scripts/run_synthetic_sweep.py --suite main --output results/synthetic_main
python scripts/run_synthetic_sweep.py --suite main --output results/synthetic_main --execute --nproc 4

# Full synthetic and language designs (plan first):
python scripts/run_synthetic_sweep.py --suite full --output results/synthetic_full
python scripts/download_data.py
python scripts/run_language_sweep.py --suite full --output results/language_full

# Execute full designs:
python scripts/run_synthetic_sweep.py --suite full --output results/synthetic_full --execute --nproc 4
python scripts/run_language_sweep.py --suite full --output results/language_full --execute --nproc 4

# Focused solver comparison:
python scripts/run_synthetic_sweep.py --suite integration --tasks associative_recall \
  --lengths 64 --seeds 0 1 2 --output results/solver_comparison --execute --nproc 4

python scripts/analyze.py --root results/synthetic_full
python scripts/analyze.py --root results/language_full
```

`--suite` also accepts `state`, `conditioning`, and `stability`; axes are documented in `configs/ablations.yaml`. The full suite runs main models at every requested training length and other ablations at the base config training length. This avoids repeating all unrelated ablations at all lengths while preserving every requested comparison. Integration excludes augmented duplication; its equivalence is already tested. Residual controls use Euler at every K. Explicit `integration` suites use all lengths supplied by the caller.

Main comparisons match exact counts within ±10% of the GRU reference by changing generic field width, recurrent state width for GRU/LSTM/structured, or Transformer FF width. The actual resolved config is saved. Impossible matching raises an error. Recorded numerical training failures remain in the summary CSV and the sweep continues; implementation errors stop execution. Nonfinite extrapolation points are explicitly marked as failures in extrapolation.json. Structured models may therefore have substantially larger memory; report both matched-capacity and fixed-state experiments. State-size and structured-conditioning sweeps intentionally keep specified dimensions and report unmatched capacities, because those are the manipulated variables. Do not mix those points into a parameter-controlled claim.

For high NFE/long lengths, the sweep planner bounds microbatch activation scale using `batch × length × NFE <= 128 × base_batch × base_length` where feasible. It chooses a divisor of the effective batch and increases accumulation, saving the actual choices in YAML. The default matrix keeps the same microbatch across K/solver comparisons; larger custom workloads can trigger this guard. This preserves optimizer-step budget, effective batch, and training examples. It is a planning heuristic, not a VRAM guarantee. `--limit N` executes/plans the first N unique runs. Sweep restarts skip completed training/eval/benchmark artifacts and resume intermediate `final.pt` checkpoints.

## Saved artifacts and resuming

Each trained run contains:

```text
config.yaml                 resolved architecture and hyperparameters
metadata.json               seed, exact parameters, global batch, world size,
                            git commit/dirty flag, Torch/CUDA/Python/GPU, data metadata
source_hashes.json          SHA256 of project Python source/tests at run start
source_snapshot.zip         exact source/tests/setup files for replaying future runs
metrics.csv                 one row per optimizer step
best.pt / final.pt          model, optimizer, AMP scaler, step, config, vocabulary,
                            best validation loss and per-rank RNG states
validation.json             final validation metrics
extrapolation.json          best-checkpoint quality at requested lengths
benchmark.json              synchronized single-device benchmark and FLOPs estimate
dynamics/                   CSV/JSON/full-state tensors for selected examples
plots/                      per-run language and dynamics plots
```

Checkpoint loads are for trusted local files only. For an interrupted run use its original config:

```bash
python scripts/train.py --config results/recall_node_heun4/config.yaml \
  --set resume=results/recall_node_heun4/final.pt
```

Exact resume requires unchanged experiment/data/world size and includes optimizer/scaler/RNG state. Configs may change output location, device, or diagnostics only. Checkpoints are atomic. A fresh output directory avoids duplicated metrics when intentionally branching from an earlier checkpoint. Resuming an already completed final checkpoint makes no additional updates.

## Plots and evidence

`python scripts/analyze.py --root <sweep>` writes one `summary/summary.csv` with all validated experiments and an `evidence.md` interpretation checklist. It never fills in missing experiments. Comparison figures separate task and training length; runs with different capacities or ablations remain distinct observations. Extrapolation lines aggregate identical configurations across seeds with ±1 sample standard deviation, not a confidence interval.

1. `quality_vs_sequence_length.png`: masked synthetic accuracy or language BPC versus held-out length.
2. `language_learning_curve.png`: train/validation CE and validation BPC versus optimizer step.
3. `quality_vs_K.png`: validation accuracy/BPC versus internal integration depth.
4. `quality_vs_wallclock.png` and `quality_vs_cost_to_best.png`: quality versus total training time and cost to selected checkpoint; validation/checkpoint IO time excluded.
5. `throughput_vs_K.png`: synchronized training throughput versus depth.
6. `hidden_norm_tokens.png`: recurrent memory norm versus token position.
7. `internal_trajectory.png` and `internal_state_pca.png`: state/derivative norms versus internal time and 2D PCA of full internal trajectories. Projected models record latent s during the solve and memory h after Q.
8. `timescales.png`: log10(-1/A) distribution over sampled token/channel pairs with negative A.
9. `parameter_comparison.png` and `quality_vs_parameters.png`: exact capacity and quality/capacity tradeoff.
10. `peak_memory_comparison.png`: peak **allocated** training memory; CUDA reserved/context memory and other processes are excluded.

Transformer update norm is activation displacement from the bare embedding, not a recurrent memory update.

Additional plots: quality versus NFE/throughput and token-field cosine heatmaps at zero and seeded random fixed states. Transformer ODE diagnostics and timescale plots are inapplicable and omitted. Structured diagnostics save A, exp(A), time constants, and the solver stability polynomial R(A/K)^K; continuous negative eigenvalues can still yield a discretely unstable update. Internal probes run in FP32 and add extra field evaluations outside timing. Full sampled state tensors are saved for further analysis.

Dominant forward FLOPs count dense multiply-adds from actual executed Linear shapes plus fused GRU/LSTM and Transformer attention estimates. Elementwise functions, normalization, softmax, optimizer work, and backward FLOPs are excluded, so these are estimates rather than profiler instruction counts. NFE=0 means not applicable to a baseline. Benchmark training includes forward/backward/clipping/AdamW on a resident batch with learning rate zero; training CSV timings also include CPU batch generation and host/device copies.

## Runtime, memory, and scientific limits

Use a resource probe before scaling a solver:

```bash
python scripts/benchmark.py --config configs/language.yaml --set model=conditioned \
  steps_per_token=4 integrator=heun --batch-size 2 --warmup 2 --repeats 5
```

See `docs/verification.md` for measured RTX 2080 Ti results. A useful estimate is:

```text
wall seconds ≈ optimizer_steps × (effective_per_rank_batch × sequence_length)
               / measured_per_rank_training_tokens_per_second
```

Measure with the actual microbatch and solver; small batches can be strongly limited by Python and kernel-launch overhead. Four-GPU DDP does not make a fixed **per-rank** workload four times faster; it processes four times as many examples per update, plus communication overhead. Do not extrapolate short-window throughput blindly to K=8/RK4 or length 2048. The default full plans contain **1,044 synthetic and 294 language runs**, sequentially using all requested GPUs for each run. At the reference training budgets this can take weeks to months, especially for high-NFE language models. Execute the main suite and focused solver comparison first.

For a tractable first pass, `synthetic_compact.yaml` and `language_compact.yaml` reduce embedding/state/field dimensions to 32/64/64, updates to 500, and validation batches to four. Language accumulation is two. They preserve all three synthetic tasks, three seeds, every solver/depth/conditioning/stability comparison, requested state-size ablations (128/256/512), and length evaluation through 2048. These are explicit scale reductions, not evidence of convergence; inspect learning curves and extend budgets before concluding a hypothesis is false. For example:

```bash
python scripts/run_synthetic_sweep.py --config configs/synthetic_compact.yaml \
  --suite full --output results/compact_synthetic --execute --nproc 4
python scripts/run_language_sweep.py --config configs/language_compact.yaml \
  --suite full --output results/compact_language --execute --nproc 4
```

Saved plans make the scope explicit.

No full research sweep has been completed as part of implementation. The delivered short comparison demonstrates the pipeline, and the overfit tests demonstrate memorization. Neither supports a claim that NODEs outperform baselines. A–G remain empirical questions until matched, three-seed runs finish. For H, the Euler version demonstrably adds no transition beyond the shared-weight residual cell; whether higher-order integration helps remains open. `refine_solver.py` additionally tests the same trained field under finer/alternative solvers, which helps distinguish learned continuous dynamics from dependence on a training discretization.

Further concerns: LayerNorm makes the generic field smooth with epsilon regularization but does not guarantee stable recurrent dynamics. Damping and negative diagonal A do not constrain learned P/Q amplification. Zero-state cosine is reported as zero by Torch convention. Time constants are in internal pseudo-time units, not physical seconds; converting them to token-memory lengths requires accounting for token-dependent coefficients. The three synthetic generators intentionally provide simple, unambiguous task variants, not a reproduction of any external benchmark protocol. Report this when comparing to published results.

Token-dependent fields may already differ at random initialization. Cosine/L2 diagnostics show dependence on x; they do not by themselves demonstrate learned, task-useful selective memory.

The native GRU/LSTM cells return FP16 state under CUDA AMP, while ODE accumulators and projected memory explicitly stay FP32. Disable AMP for a comparison sensitive to state quantization. The three ODE conditioning formulations share FP32 state accumulation to avoid this confound within H1.

## Executed three-seed follow-up

The focused follow-up compares conditioned, autonomous projected, structured, and GRU models at matched capacity, then compares conditioned Euler/K=1, Euler/K=4, and Heun/K=4. Euler/K=1 is shared, giving 18 unique runs for three seeds.

```bash
python scripts/run_focused_comparison.py --output results/focused_recall
python scripts/run_focused_comparison.py --output results/focused_recall --execute --gpus 0 1 2 3
```

Defaults are in `configs/focused_recall.yaml`: length 64, embedding 32, reference memory/latent 64, batch 64, 1,000 attempted updates, and 512 validation/test examples per seed and length. Independent single-GPU jobs are queued over four GPUs, so every condition has the same global batch without compact-model DDP communication overhead. The train/validation streams are paired across conditions within each seed. Checkpoint selection uses validation CE; final test evaluation adds a sampling-seed offset of 10^12 to use an independent synthetic stream. The language loader still uses its original validation text when this optional offset is used.

All training completes before isolated benchmarks run sequentially on a common GPU. Benchmarks initialize AMP scaling from the trained checkpoint and report skipped updates. Outputs include `summary/test_summary.csv`, `seed_aggregates.csv`, `paired_contrasts.json`, `shortcut_baselines.csv`, and `focused_report.md`, plus seed error bars, learning curves, quality/compute and length-extrapolation plots. The frequent-value and recent-value baselines test whether a model exploits a shortcut rather than retrieving the queried key. Exploratory paired 95% t intervals use three seed differences and are not corrected for multiple comparisons.
