# Notebooks

Open a notebook in your editor, select the project's Python environment as its
kernel, and run the cells in order. Commands and paths assume the repository root;
the loss notebook also detects when the kernel starts inside `notebooks/`.

| Notebook | Use |
| --- | --- |
| [01 — BPE](01_bpe_fineweb2.ipynb) | Train and inspect a tokenizer |
| [02 — Generation](02_mlx_generation.ipynb) | Sample a saved MLX model |
| [03 — Loss curves](03_compare_run_losses.ipynb) | Compare runs; rerun the final cell to refresh |
| [04 — Attention](04_attention_variants.ipynb) | Compare full and sliding-window masks |
| [05 — Compilation](05_compile_numerics.ipynb) | Benchmark native, corrected, and compiled math |
| [06 — Training statistics](06_training_statistics.ipynb) | Compare LR/clip runs; inspect gradients, noise, activations, and logits |
| [07 — Memorization](07_memorization.ipynb) | Overfit a Russian passage; check continuation and checkpoint reload |
| [08 — μP coordinate check](08_mup_coordinate_check.ipynb) | Compare activation and update scales before attempting μTransfer |
| [09 — μP sweep results](09_mup_sweep_results.ipynb) | Follow the LR and decay search and compare the proxy and target results |

Install the extras needed by your notebook:

```sh
python -m pip install -e '.[pretraining-mlx,analysis]'
```

For notebook 02, set `run_dir` to a completed run. For notebook 03,
set `selected_names` to the run folders you want, or leave it empty for all runs.
Notebook 06 opens with the baseline/double-LR analysis and saved graphs; edit
`run_names` to select another matched pair. Rerunning needs only the analysis extra.
Follow-up sections read `clip_name`, `accum_name`, and `accum8_name` for the clip-5,
×4, and ×8 runs. Accumulation comparisons use processed tokens rather than raw update counts.
The final section compares the width-256/native-softmax run with the previous best.
Notebook 07 contains executed results; rerunning trains fresh models on Metal and saves
separate `artifacts/pretraining/memorization-*` directories.
Notebook 08 runs a small synthetic width check in the MLX environment. The
proxy LR sweep is a separate command in `experiments/mup_transfer/README.md`.
Notebook 09 includes saved plots and reruns with only the analysis extra.
