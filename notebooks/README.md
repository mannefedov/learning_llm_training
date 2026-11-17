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

Install the extras needed by your notebook:

```sh
python -m pip install -e '.[pretraining-mlx,analysis]'
```

For notebook 02, set `run_dir` to a completed run. For notebook 03,
set `selected_names` to the run folders you want, or leave it empty for all runs.
Notebook 06 opens with the baseline/double-LR analysis and saved graphs; edit
`run_names` to select another matched pair. Rerunning needs only the analysis extra.
Follow-up sections read `clip_name` and `accum_name` for the clip-5 and accumulation
runs. Accumulation comparisons use processed tokens rather than raw update counts.
