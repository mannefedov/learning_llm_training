# Larger batches during annealing

Requires the project MLX environment, the prepared fresh-data cache, and
the step-100,000 checkpoint from `gpt-mlx-mup-width256-2b-03`.
Run from the repository root.

```sh
bash experiments/annealing/run_batch_comparison.sh
```

Compare the three results with the completed 1,000-update fresh-data anneal
(validation loss 3.71462). All trials use its same 20.48M-target prefix, saved
starting LR, and cosine decay to zero. Adam moments and the step counter carry
over from the source checkpoint.

| Trial | Accumulation | Effective sequences | Updates |
| --- | ---: | ---: | ---: |
| Existing baseline | 8 | 80 | 1,000 |
| 2× batch | 16 | 160 | 500 |
| 4× batch | 32 | 320 | 250 |
| 8× batch | 64 | 640 | 125 |

Progress reports loss, LR, and throughput. Results, diagnostics, samples, and
checkpoints go to `artifacts/pretraining/gpt-mlx-mup-width256-fresh-batch*-`
directories with an anneal suffix. Evaluation/diagnostic intervals approximately
match the baseline's token spacing, with final measurements at the exact budget.
Noise probes use the trial's effective batch size, so compare both their variance
and signal estimates when interpreting ratios.

To resume an interrupted trial, use its saved config and checkpoint with
`--resume`. Start the script once to avoid repeating completed trials.

## Staircase cooldowns

Run three fresh-cache branches at the original effective batch of 80:

```sh
bash experiments/annealing/run_staircase_comparison.sh
```

The 1k, 3k and 5k budgets each hold 100%, 50%, 25% and 10% of the saved
base LR over successive quarters, reaching exactly zero on the final update.
Configs live in `configs/pretraining/sequential_gpt_mlx_mup_staircase_*.toml`.
This four-level schedule is our experimental choice. Compare with cosine
branches at matching token budgets and the constant-LR control. The completed
cosine grid contains 1k and 3k runs, with 6k as the nearest existing run to 5k.
Saved branch configs preserve the staircase schedule on interrupted resume.
