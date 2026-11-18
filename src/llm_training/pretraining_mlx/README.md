# MLX pretraining

Run from the repository root on an Apple silicon Mac, with your environment activated.

## Train

[Train a tokenizer](../tokenization/README.md) first and set `[tokenizer].path`
in `configs/pretraining/tiny_gpt_mlx.toml` to its saved JSON.

```sh
python -m pip install -e '.[pretraining-mlx]'
python -m llm_training.pretraining_mlx --config configs/pretraining/tiny_gpt_mlx.toml
```

Edit `[model]` for dimensions and `[training]` for steps, batch size, and learning
rate. A quick check uses `steps = 5` and `eval_batches = 1`.
`device = "auto"` selects Metal when available; `"gpu"` and `"cpu"` are explicit choices.

## Sequential training with a disk cache

Use this config for a fixed 200,000-step budget without random window sampling:

```sh
python -m llm_training.pretraining_mlx.prepare --config configs/pretraining/sequential_gpt_mlx.toml
python -m llm_training.pretraining_mlx --config configs/pretraining/sequential_gpt_mlx.toml
```

Preparation streams documents in order, stopping at the document limit or
`steps × batch_size × gradient_accumulation_steps × context_length + 1` token IDs,
whichever comes first (`gradient_accumulation_steps` defaults to 1).
The last document can be cut short. A smaller corpus is reused across epochs. At 200k steps, batch 10, context
256, the training token file is about 2.05 GB (1.91 GiB), plus validation.

When preparation prints `Cache ready`, the cache is complete.

Training reads sequential batches from memory-mapped files. After each full pass,
it restarts at the beginning until the total `steps` target is reached.
Validation keeps fixed random windows from its separate cached split. Repeated
runs reuse the cache without fetching the corpus again. The manifest records the
actual document/token counts and tokenizer identity. If preparation is interrupted,
choose a new `[cache].directory`; incomplete caches are not reused.

## Samples during training

The optional `[generation]` table is enabled by default:

```toml
[generation]
enabled = true
every = 500
prompts = ["Сегодня утром я", "В Москве сегодня"]
max_new_tokens = 100
temperature = 0.8
seed = 42
```

Samples are written at step zero, every configured interval, and the final step.
Set `enabled = false` to disable. Follow the printed run path:

```sh
tail -f artifacts/pretraining/tiny-gpt-mlx-run-01/samples.txt
```

## Generate after training

Use your completed run directory:

```sh
python -m llm_training.pretraining_mlx.generate \
  --run-dir artifacts/pretraining/tiny-gpt-mlx-run-01 \
  --prompt "Сегодня утром я" \
  --temperature 0.5 0.8 1.0 --max-new-tokens 100 --seed 42
```

Repeat `--prompt` for more sentence beginnings. Generation stops at EOS or the
token limit. Explore interactively in [the generation notebook](../../../notebooks/02_mlx_generation.ipynb).

## Outputs and curves

The run directory contains metrics, metadata, config/tokenizer copies, sample
text, and `checkpoint.json` pointing to the latest saved state in `checkpoints/`.

## Collect training statistics

For the basic 256-token, four-layer full-attention experiment, reuse the prepared
cache and run in the standard MLX environment:

```sh
.venv/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_diagnostics.toml
```

Outputs go under `artifacts/pretraining/gpt-mlx-basic-diagnostics-01` (numbered
automatically if occupied). Model, schedule and reporting intervals are in the config.

Enable collection in any MLX config with:

```toml
[statistics]
every = 1000
probe_batches = 2
```

`statistics.jsonl` contains one JSON record at initialization/resume, update 1,
every configured interval, and the final update. `every = 0` disables it (default).
Its cadence is independent of `[training].eval_every`; increasing either interval
reduces its own reporting cost. Each record contains:

- Interval mean/max of the global **pre-clipping** gradient norm and fraction of
  updates clipped. Intervals restart on resume and after each statistics record.
- Per-tensor shape, element count, preupdate weight RMS, raw gradient RMS, actual
  update RMS, and update/weight RMS ratio. Updates include Adam and weight decay;
  gradients are from the sampled training batch before clipping. Ratios are null
  for near-zero weights; inspect absolute update RMS for those tensors.
- Fixed training-corpus and held-out probe losses, each averaged over
  `probe_batches` batches. Windows use independent fixed seeds, so probes do not
  alter training sampling. They are separate from the validation in `metrics.csv`.
- `measurement_seconds`: sampled snapshot/reduction/probe time, excluding file
  writing and the small per-update norm collection cost.

The initial record has current weight RMS and probe losses but no gradient/update
measurements. Collection stores scalar summaries, not full tensors or histories.
Parameter snapshots exist in memory only for sampled updates. Read a record with:

```sh
tail -n 1 artifacts/pretraining/gpt-mlx-basic-diagnostics-01/statistics.jsonl \
  | python -m json.tool
```

Open [notebook 06](../../../notebooks/06_training_statistics.ipynb) to compare
losses, clipping, per-layer gradients/updates, and parameter scales across the
baseline and double-LR runs.

### Clip-threshold experiment and extended probes

Start a fresh matched baseline-LR run with `grad_clip = 5` and extended diagnostics:

```sh
.venv/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_diagnostics_clip5.toml
```

Results go under `artifacts/pretraining/gpt-mlx-basic-diagnostics-clip5-01`.
Only the clipping threshold changes training settings. The added measurements are
optional and do not perform optimizer updates:

```toml
[statistics]
every = 1000
probe_batches = 2
activations = true
logits = true
gradient_noise_batches = 4
gradient_noise_every = 5000
```

New records use schema version 2; existing keys are retained. Extra fields are:

- `probes.activations`: mean, standard deviation, RMS and absolute maximum at
  embeddings, each block's normalized inputs/branch outputs/residuals, and final
  normalization. These use the first fixed training-probe batch at current weights.
- `probes.logits`: raw logit summaries, mean standard deviation across vocabulary,
  predictive entropy (nats), maximum/target probabilities and top-1 token accuracy
  on that same batch.
- `gradient_noise`: at fixed current weights, compute raw gradients on separately
  seeded training batches. Reports mean-gradient norm, batch-gradient variance,
  mean pairwise cosine similarity, estimated noise/signal ratio and noise scale
  in tokens. No clipping or Adam transformation is applied to these gradients.

Noise probes run at initialization, update 1, every `gradient_noise_every` and
the final step; the interval must be a multiple of `statistics.every`. Four batches
mean four extra backward passes per noise report (168 across this config's run).
Their time is recorded separately and included in total `measurement_seconds`.
The same probe windows are reused across reports, independently of training RNG.

For gradients `g_i`, let `S = sum(||g_i - mean(g)||²)/(K-1)`. The squared signal
estimate is `||mean(g)||² - S/K`; `noise_scale_tokens` is batch tokens times
`S/signal`. If signal is nonpositive, ratios are null rather than forced positive.
This is a small-sample diagnostic, not an exact optimal batch size or an Adam-aware
noise estimate; tokens within a context window are correlated. See the
[gradient-noise-scale paper](https://arxiv.org/abs/1812.06162).

The new options default to off (`gradient_noise_batches = 0`). Older runs lack
these measurements and remain usable for their existing loss/gradient/update plots.

### Gradient accumulation

Start the matched 512M-token experiment with four microbatches per update:

```sh
.venv/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_diagnostics_accum4.toml
```

Results go under `artifacts/pretraining/gpt-mlx-basic-diagnostics-accum4-01`.
`batch_size = 10` remains the microbatch size. With
`gradient_accumulation_steps = 4`, the effective batch is 40 sequences / 10,240
tokens. Gradients and loss are averaged across the four microbatches at unchanged
weights, then clipped once before one AdamW update. Only one microbatch's
forward/backward graph is kept at a time.

`steps`, LR schedules, evaluations, checkpoints, samples and statistics intervals
count **optimizer updates**. This config uses 50k updates and 500 warmup updates,
matching the baseline's total tokens and warmup/decay curve by tokens. LR remains
`3e-4` to `3e-5`, clipping returns to 1. Validation runs every 125 updates (every
other baseline evaluation by tokens); statistics every 250 and noise every 1,250.
Adam betas and weight decay remain unchanged per update, with fewer total updates.

Validation and fixed forward probes keep the microbatch size, preserving their
windows. Each noise-probe gradient averages a full effective batch, using the
same microbatch accumulation; its `tokens_per_batch` reports that effective size.
Four noise batches now require 16 backward passes per report.

Sequential training packs a complete effective batch before splitting it into
microbatches. Epoch tails smaller than one effective batch are omitted. Progress,
throughput and preparation budgets count all microbatches. Resume checkpoints are
written only after complete optimizer updates and require the same accumulation
factor; old checkpoints imply factor 1. Compare runs by tokens, not raw step number.

### Larger-data width-256 experiment

Prepare the larger cache, then start a fresh run:

```sh
.venv/bin/python -m llm_training.pretraining_mlx.prepare \
  --config configs/pretraining/sequential_gpt_mlx_diagnostics_width256_2b.toml
.venv/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_diagnostics_width256_2b.toml
```

Wait for `Cache ready` before training. The cache is
`artifacts/data/fineweb2-rus-2b-tokens`; results go under
`artifacts/pretraining/gpt-mlx-basic-diagnostics-width256-2b-01`.

## Compile forward/backward

```toml
[training]
compile_step = true
```

This compiles model forward/backward together. Optimizer updates, validation,
sampling, and file writes stay outside. The first batch includes compilation
work. Defaults to `false`; it can be changed on resume.

The corrected-math compiled config is
`configs/pretraining/sequential_gpt_mlx_swa_3to1_stable_compiled.toml`.
For timings, use [notebook 05](../../../notebooks/05_compile_numerics.ipynb).

## Learning-rate schedule

For a fresh 4K SWA run with linear warmup and cosine decay, use the patched
MLX environment and the existing prepared cache:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_swa_fused_4k_cosine.toml
```

The schedule settings live in `[training]`:

```toml
lr_schedule = "cosine"
learning_rate = 0.0003
min_learning_rate = 0.00003
warmup_steps = 1250
lr_decay_steps = 125000
```

Update 1 uses `learning_rate / warmup_steps`; the last warmup update reaches
the peak. Cosine decay reaches the minimum at `lr_decay_steps` and stays there.
Set `warmup_steps = 0` to start at the peak. Without `lr_decay_steps`, decay ends
at `training.steps`. Existing configs default to constant LR.

On resume, the schedule uses the absolute saved step. Keep an explicit
`lr_decay_steps` unchanged when extending `training.steps` to preserve the curve.
Changing schedule settings deliberately replans the curve at that saved step;
it does not restart warmup. `metrics.csv` records the LR used for each reported
update (empty at initialization), and progress displays `lr`.

## Checkpoint and resume

```toml
[checkpoint]
every = 1000
keep = 1
```

The latest checkpoint replaces the previous one by default. Set `keep = 3` to
retain three per run, or `every = 0` to save only at the final step. Saving briefly
needs space for one extra checkpoint; older copies are removed after success.

Resume using the original run directory (or a retained `checkpoint.json`):

```sh
python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_swa_1to1.toml \
  --resume artifacts/pretraining/gpt-mlx-swa-1to1-512m-01
```

`steps` is the total target, not additional steps. To do a second pass over a
200k-step cache, set `steps = 400000` and resume the completed run; keep the same
cache and skip preparation. Resume automatically finds the epoch and batch from
the saved step. A tail smaller than one full batch is omitted each epoch.

Resume creates a fresh numbered
run for logs, keeping the source run intact. Model, tokenizer, data, batch size,
and optimizer settings other than LR/schedule settings must match. With
`lr_schedule = "constant"`, set `[training].learning_rate` to use a lower rate; Adam
moments and the update count are restored. Only checkpoints written with resume support
can resume; earlier checkpoints still work for generation.

For loss plots, install `.[analysis]` and open
[the comparison notebook](../../../notebooks/03_compare_run_losses.ipynb).

For a fused full-attention experiment, use the
[model instructions](../models/README.md#fused-full-attention-mlx).

## Experimental fused SWA at 4K

Use the [locally patched MLX build](../../../experiments/mlx_swa/README.md):

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_swa_fused_4k.toml
```

Reuses the existing token cache and starts a new run under
`artifacts/pretraining/gpt-mlx-swa-fused-4k-512m-01`. The config selects the
experimental `mlx_fused_window` backend for local layers and fused causal
attention for full layers. It uses native softmax; corrected softmax is incompatible.
The standard `.venv` does not contain this kernel patch.

Inspect progress timing, validation loss, and the generated samples. Longer
prompt continuations are more useful for assessing the added context than the
short prompts alone. This run changes context and model size; to compare kernel
speed alone, keep the config settings and change only `attention_backend` to
`reference` in a separate config/run.

Progress shows `tok/s`: average training tokens per second since this start/resume,
including validation, sampling, checkpointing, and compilation time where applicable.
`train` tracks the latest batch each step. The status refreshes in place about
once per second in a terminal and wraps at the current width. Redirected
output adds a plain status line about every 30 seconds.
`val` and CSV loss rows update at `eval_every`.
`metrics.csv` records `tokens_per_second`, `tokens_processed`, and `elapsed_seconds`
at each loss evaluation. Each update counts
`batch_size × gradient_accumulation_steps × context_length` tokens.
