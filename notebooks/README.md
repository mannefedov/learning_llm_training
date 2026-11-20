# Notebooks

Open [the transformer explorer](transformer-explorer.html) in a browser to
adjust model dimensions and inspect layers, attention heads, tensor shapes,
and parameter counts. It includes the current RoPE and learned-position presets.

For zoomable attention maps, start the
[local attention viewer](../experiments/attention/README.md). Notebook 11
contains its static overviews and interpretation guides.

Open a notebook in your editor, select the project's Python environment as its
kernel, and run the cells in order. Commands and paths assume the repository root;
the loss notebook also detects when the kernel starts inside `notebooks/`.

| Notebook | Use |
| --- | --- |
| [01 — BPE](01_bpe_fineweb2.ipynb) | Train and inspect a tokenizer |
| [02 — Generation](02_mlx_generation.ipynb) | Sample a saved MLX model |
| [03 — Loss curves](03_compare_run_losses.ipynb) | Compare runs; rerun the final cell to refresh |
| [08 — μP coordinate check](08_mup_coordinate_check.ipynb) | Compare activation and update scales before attempting μTransfer |
| [09 — μP sweep results](09_mup_sweep_results.ipynb) | Follow the LR and decay search and compare the proxy and target results |
| [10 — Anneal branches](10_anneal_branches.ipynb) | Compare six fresh-data cooldowns against constant LR at equal steps |
| [11 — Attention sinks](11_attention_sinks.ipynb) | Inspect first-position preference across layers and heads in saved models |
| [12 — Mixture of experts](12_moe_tutorial.ipynb) | Follow routing, expert mixing and balancing, then train a tiny MoE |
| [13 — Latent attention](13_mla_tutorial.ipynb) | Build MLA projections, decoupled RoPE and compressed-cache decoding, then train a tiny GPT |
| [14 — Einsum](14_einsum_tutorial.ipynb) | Build index notation from small arrays through attention and the MLA contractions |

Install the extras needed by your notebook:

```sh
python -m pip install -e '.[pretraining-mlx,analysis]'
```

For notebook 02, set `run_dir` to a completed run. For notebook 03,
set `selected_names` to the run folders you want, or leave it empty for all runs.
Notebook 08 runs a small synthetic width check in the MLX environment. The
proxy LR sweep is a separate command in `experiments/mup_transfer/README.md`.
Notebook 09 includes saved plots and reruns with only the analysis extra.
