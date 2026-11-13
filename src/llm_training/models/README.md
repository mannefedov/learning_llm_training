# GPT models

| File | Backend | Run it |
| --- | --- | --- |
| `gpt.py` | PyTorch | [Training and sampling](../pretraining/README.md) |
| `gpt_mlx.py` | MLX | [Training and sampling](../pretraining_mlx/README.md) |

Both implement a small GPT with causal attention, learned positions, pre-norm
blocks, GELU, and tied embeddings. Dimensions come from `[model]` in the training config.

Architecture reference: [build-nanogpt](https://github.com/karpathy/build-nanogpt).

## Attention variants

Set one type per layer in `[model]` (omitting the list keeps full attention):

```toml
attention_types = ["sliding_window", "full", "sliding_window", "full"]
sliding_window = 64
```

The window includes the current token. At context 256, start with the 1:1 config:

```sh
python -m llm_training.pretraining_mlx --config configs/pretraining/sequential_gpt_mlx_swa_1to1.toml
```

Use `sequential_gpt_mlx_swa_3to1.toml` to compare three local layers and one full
layer. Both reuse the sequential token cache; prepare it using the
[cache instructions](../pretraining_mlx/README.md#sequential-training-with-a-disk-cache) if needed.

Gemma 2 uses 1:1; Gemma 3 uses 5:1. For this four-layer model, 1:1 and 3:1 are
starting experiments. See [the attention notebook](../../../notebooks/04_attention_variants.ipynb)
for sources and masks.

`attention.py` and `attention_mlx.py` share projections within each backend;
`build_attention` selects the variant. SWA tiles the score computation; no KV
cache or custom kernels are implemented. Speed has not been benchmarked.
