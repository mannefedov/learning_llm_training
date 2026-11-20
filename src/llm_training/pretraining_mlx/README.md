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
prompts = ["Сегодня утром я", "В Москве сегодня", "Чтобы приготовить суп,", "Учёные обнаружили", "— Ты куда"]
max_new_tokens = 100
temperatures = [0.0, 0.3, 0.5, 0.8]
seed = 42
```

Samples are written at step zero, every configured interval, and the final step.
Temperature `0` uses greedy argmax decoding. Each prompt is sampled at every
listed temperature, giving 20 labeled continuations per event with these defaults.
The legacy single `temperature` setting is also supported; use one setting or
the other. Sampling uses local random keys and preserves the training RNG.
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
  --temperature 0 0.3 0.5 0.8 --max-new-tokens 100 --seed 42
```

Repeat `--prompt` for more sentence beginnings. Generation stops at EOS or the
token limit. Explore interactively in [the generation notebook](../../../notebooks/02_mlx_generation.ipynb).

## Outputs and curves

The run directory contains metrics, metadata, config/tokenizer copies, sample
text, and `checkpoint.json` pointing to the latest saved state in `checkpoints/`.

## Collect training statistics

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
tail -n 1 path/to/run/statistics.jsonl \
  | python -m json.tool
```

Enable activation and logit summaries with `activations = true` and
`logits = true` in `[statistics]`. Set `gradient_noise_batches` to at least 2
to estimate raw gradient noise. Set `gradient_noise_every` to a multiple of
`every` to control the extra backward probes. See the
[estimator notes](../../../docs/pretraining/gradient_noise.md).

Set `gradient_accumulation_steps` in `[training]` to combine microbatches
before one optimizer update. Match training targets and retune LR when
changing the effective batch.
[Past comparisons](../../../docs/models/experiment_findings.md)

## Compile forward/backward

```toml
[training]
compile_step = true
```

This compiles model forward/backward together. Optimizer updates, validation,
sampling, and file writes stay outside. The first batch includes compilation
work. Defaults to `false`; it can be changed on resume.

## Learning-rate schedule

Set the schedule in `[training]`:

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

To keep warmup but disable end decay, set `lr_decay = false` in `[training]`.
The LR then stays at its peak, even when extending `steps` on resume. Existing
configs retain decay unless explicitly disabled. `--anneal-steps` enables decay
for that branch regardless of this setting.

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

Set `keep = 3` to retain three recent checkpoints, or `every = 0` for final-only
periodic storage. The trainer also saves the exact last peak-LR update before
decay and protects it from pruning. `stable-checkpoint.json` points to this
resumable checkpoint; pass that file to `--resume`. With decay disabled, the
pointer follows the latest saved checkpoint after warmup. A protected checkpoint
can consume one extra model/optimizer copy beyond `keep`. Saving briefly needs
space for another copy; older copies are removed after success.

Resume using the original run directory (or a retained `checkpoint.json`):

```sh
python -m llm_training.pretraining_mlx \
  --config path/to/resume-config.toml \
  --resume path/to/original-run
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

### Anneal on resume

Pass `--anneal-steps N` with `--resume` to run N additional updates (N ≥ 2).
The first update uses the saved optimizer LR and cosine decay reaches zero on
the last update. This overrides the config's step budget and LR schedule for
the branch. The source checkpoint supplies the model, optimizer state, and
batch position. Use the same source checkpoint for each branch.

```sh
run=artifacts/pretraining/gpt-mlx-mup-width256-2b-03
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config "$run/config.toml" --resume "$run" --anneal-steps 1000
```

The trainer creates a numbered directory with `-anneal-N-` in its name.
To continue an interrupted anneal, pass the branch's saved `config.toml`
and branch directory with `--resume`. Its saved schedule picks up at the
checkpoint step. Use `--anneal-steps` when starting a new branch.

### Compare annealing with constant LR on fresh data

The config below prepares 409.6M training targets after the original 2B
cache and copies its validation tokens. Each branch starts at batch zero of
the new cache with the same weights and Adam state from step 100,000.

```sh
config=configs/pretraining/sequential_gpt_mlx_mup_fresh_constant_control.toml
PYTHONPATH=src .venv/bin/python -m llm_training.pretraining_mlx.prepare --config "$config"
```

After preparation, run the control and six anneals sequentially:

```sh
config=configs/pretraining/sequential_gpt_mlx_mup_fresh_constant_control.toml
run=artifacts/pretraining/gpt-mlx-mup-width256-2b-03
for n in 0 1000 3000 6000 10000 15000 20000; do
  extra=()
  if [ "$n" -gt 0 ]; then extra=(--anneal-steps "$n"); fi
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
    .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
    --config "$config" --resume "$run" --resume-new-cache "${extra[@]}" || break
done
```

The control stays at the saved base LR for 20,000 updates. Compare each
anneal's final validation loss with the control at that same step. To resume
an interrupted branch, use its saved config and checkpoint with `--resume`.
The checkpoint retains the new cache's batch offset.

## Throughput and progress

Progress shows `tok/s`: average training tokens per second since this start/resume,
including validation, sampling, checkpointing, and compilation time where applicable.
`tok/param` is total training targets divided by total model parameters, including
embeddings and all MoE experts. It carries forward on resume; evaluation and
generated tokens are excluded. New checkpoints retain the cumulative count
across effective-batch changes. Older checkpoints infer it from saved steps
and effective batch.
`train` tracks the latest batch each step. The status refreshes in place about
once per second in a terminal and wraps at the current width. Redirected
output adds a plain status line about every 30 seconds.
`val` and CSV loss rows update at `eval_every`.
`metrics.csv` records `tokens_per_second`, `tokens_processed`, and `elapsed_seconds`
at each loss evaluation. Each update counts
`batch_size × gradient_accumulation_steps × context_length` tokens.
