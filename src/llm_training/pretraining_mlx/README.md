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
`steps × batch_size × context_length + 1` token IDs, whichever comes first.
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
and optimizer settings must match. Only checkpoints written with resume support
can resume; earlier checkpoints still work for generation.

For loss plots, install `.[analysis]` and open
[the comparison notebook](../../../notebooks/03_compare_run_losses.ipynb).
