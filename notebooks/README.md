# Notebooks

Open a notebook in your editor, select the project's Python environment as its
kernel, and run the cells in order. Commands and paths assume the repository root;
the loss notebook also detects when the kernel starts inside `notebooks/`.

| Notebook | Use |
| --- | --- |
| [01 — BPE](01_bpe_fineweb2.ipynb) | Train and inspect a tokenizer |
| [02 — Generation](02_mlx_generation.ipynb) | Sample a saved MLX model |
| [03 — Loss curves](03_compare_run_losses.ipynb) | Compare runs; rerun the final cell to refresh |

Install the extras needed by your notebook:

```sh
python -m pip install -e '.[pretraining-mlx,analysis]'
```

For notebook 02, set `run_dir` to a completed run. For notebook 03,
set `selected_names` to the run folders you want, or leave it empty for all runs.
