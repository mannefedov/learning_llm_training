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

## Corrected softmax experiment

Set `stable_softmax = true` in `[model]` to use corrected attention softmax and
hard-target cross-entropy in either backend. The default stays `false` for
existing runs. Start the matching 3:1 experiment with your existing token cache:

```sh
python -m llm_training.pretraining_mlx --config configs/pretraining/sequential_gpt_mlx_swa_3to1_stable.toml
```

This starts a new run; resume requires matching model settings. Implementation
and source: [numerics.py](numerics.py), [MLX version](numerics_mlx.py),
[corrected reference and erratum](https://jonbarron.info/softmaxfix/).

## Fused full attention (MLX)

```sh
python -m llm_training.pretraining_mlx --config configs/pretraining/sequential_gpt_mlx_full_fused.toml
```

Reuses the existing token cache and starts a new run under
`artifacts/pretraining/gpt-mlx-full-fused-512m-01` (numbered if needed).
This config uses full attention, two heads of size 64, native softmax, and
compiled forward/backward. It changes the head layout from the earlier runs;
compare kernel speed with `attention_backend = "reference"` and otherwise
identical settings. Existing checkpoints with a different layout cannot resume.

`attention_backend = "mlx_fused"` requires Metal and forces MLX's fused kernel;
unsupported shapes fail instead of falling back. Training sequences must exceed
eight tokens in MLX 0.32.4; shorter prefixes work for generation. It cannot be combined with
`stable_softmax = true` or SWA. The default backend remains `reference`.

[Exploration backlog](../../../docs/explorations/README.md).
