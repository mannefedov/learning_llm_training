# μTransfer experiment

This folder contains an opt-in MLX μP/μTransfer experiment. The μP model uses
an independent, zero-initialized readout; existing standard GPT runs retain
their tied embeddings and output head.

The sweep trains models on identical batches and compares peak learning rates
and matrix weight decay. Keep depth, context length, data order, effective batch,
optimizer, and schedule fixed when comparing widths.

Run the eight-update width check first:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python experiments/mup_transfer/coordinate_check.py
```

The check compares three initialization seeds at widths 64, 128 and 256.
Read `loss_improvement` and `target_logit_change_mean` alongside activation
RMS. Full-vocabulary logit RMS can decrease with width even when the
target-token response and loss improvement remain stable.

To reproduce the corrected LR comparison on the existing 512M-token cache:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python experiments/mup_transfer/sweep.py \
  --cache artifacts/data/fineweb2-rus-512m-tokens \
  --output artifacts/pretraining/mup-proxy-attention-corrected.jsonl \
  --base-width 64 --proxy-width 128 --target-width 256 \
  --batch-size 10 --gradient-accumulation-steps 8 \
  --steps 2000 --eval-every 250 --weight-decays 0.6 \
  --learning-rates 0.0136 0.0192
```

Each candidate uses the same seed, data order, warmup and cosine schedule.
The output records the transferred base LR and effective target matrix LR.
The command prints validation progress at step 0 and every 250 steps for each
candidate. It reports training tok/s for the latest interval, excluding validation
time, and saves interval rates and the overall training rate in the JSONL result.
The run skips matching completed candidates in that file, so choose a new
output filename to repeat the experiment. `mup-proxy-serious-lr.jsonl`,
`mup-proxy-serious-refine.jsonl`, `mup-proxy-serious-decay.jsonl`, and
`mup-target-serious.jsonl` used the superseded attention scale. The corrected
target comparison is in `mup-target-attention-corrected.jsonl`.

The corrected 2,000-update proxy prefers 0.0136 at decay 0.6, while width 256
prefers 0.0192 (validation 4.357 versus 4.434). These runs test roughly 41M
training targets per candidate and decay LR to 10% by step 2,000. They do not
identify the best 100,000-update/2B-token schedule or long-run decay. The
prototype does not implement the 2025 width-dependent decay rule.

## Matched four-width LR test

Start with seed 42 and every power of two from 2⁻¹³ through 2⁻³. Each LR is
exactly double the previous one. Extend the grid if a winner is at
an edge. Add intermediate points around the best region on the same grid for
every width, then repeat those points with seeds 43 and 44. Use mean final
validation loss and seed spread to compare widths. The command below uses
the same four layers, four heads, effective
batch of 80, context of 256 and 2,000 updates at every width. Weight decay is
zero to isolate the LR test from changes in effective matrix decay across widths.

```sh
for width in 32 64 128 256; do
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
    .local/venv-mlx-swa/bin/python experiments/mup_transfer/sweep.py \
    --cache artifacts/data/fineweb2-rus-512m-tokens \
    --output artifacts/pretraining/mup-width-grid-powers-of-two.jsonl \
    --base-width 64 --proxy-width "$width" --target-width 256 \
    --steps 2000 --eval-every 250 --seed 42 --weight-decays 0 \
    --learning-rates 0.0001220703125 0.000244140625 0.00048828125 \
      0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 \
      0.03125 0.0625 0.125 || break
done
```

The first pass is 44 candidates with 40.96M token presentations each. In
notebook 09, the final section plots final validation loss versus LR for each
width and individual validation trajectories versus training step. Compare
both the minima and the loss penalty of using the proxy's preferred LR at
width 256. A flat region can make neighboring LRs interchangeable.

Before choosing a 2B schedule, compare the transferred LR and its neighbors
in a longer target-width pilot with the intended data order, batch, warmup,
decay and LR schedule. A matched short sweep measures width transfer at its
own training budget. To measure whether μP improves transfer relative to
standard parameterization, run a separate standard grid with the same
experimental controls and a range that brackets its own minima.

## Width-256 run on the 2B cache

Run from the repository root with the prepared 2B cache:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_mup_width256_2b.toml
```

This starts fresh and writes metrics, samples and checkpoints to
`artifacts/pretraining/gpt-mlx-mup-width256-2b-01`. The config uses the
width-256 LR selected by the zero-decay sweep and the same token budget,
batch size and validation cadence as the standard 2B run.
