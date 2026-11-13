# GPT models

| File | Backend | Run it |
| --- | --- | --- |
| `gpt.py` | PyTorch | [Training and sampling](../pretraining/README.md) |
| `gpt_mlx.py` | MLX | [Training and sampling](../pretraining_mlx/README.md) |

Both implement a small GPT with causal attention, learned positions, pre-norm
blocks, GELU, and tied embeddings. Dimensions come from `[model]` in the training config.

Architecture reference: [build-nanogpt](https://github.com/karpathy/build-nanogpt).
