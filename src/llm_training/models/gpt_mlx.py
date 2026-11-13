"""Parallel MLX version of gpt.py; the PyTorch implementation stays independent.

MLX reference: https://ml-explore.github.io/mlx/build/html/index.html
Architecture reference: https://github.com/karpathy/build-nanogpt
"""

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.n_layers,
               self.n_heads, self.d_model) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def __call__(self, x):
        batch, length, width = x.shape
        head_size = width // self.n_heads
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = [a.reshape(batch, length, self.n_heads, head_size)
                   .transpose(0, 2, 1, 3) for a in (q, k, v)]
        scores = (q @ k.swapaxes(-2, -1)) / math.sqrt(head_size)
        causal_mask = mx.arange(length)[:, None] >= mx.arange(length)[None, :]
        scores = mx.where(causal_mask, scores, -float("inf"))
        attended = mx.softmax(scores, axis=-1) @ v
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.proj(attended)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(approx="precise"),  # MLX name for the original tanh approximation.
            nn.Linear(4 * config.d_model, config.d_model),
        )

    def __call__(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = [Block(config) for _ in range(config.n_layers)]
        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-5)

        def initialize(path, parameter):
            if parameter.ndim >= 2:
                # Smaller residual projections match gpt.py initialization.
                residual = path.endswith(("attention.proj.weight", "mlp.layers.2.weight"))
                std = 0.02 / math.sqrt(2 * config.n_layers) if residual else 0.02
                return mx.random.normal(parameter.shape) * std
            return mx.zeros_like(parameter) if path.endswith("bias") else mx.ones_like(parameter)

        self.update(tree_map_with_path(initialize, self.parameters()))

    def __call__(self, input_ids, targets=None):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.context_length:
            raise ValueError("input_ids must have shape (batch, time) within context_length")
        x = self.token_embedding(input_ids) + self.position_embedding(mx.arange(input_ids.shape[1]))
        for block in self.blocks:
            x = block(x)
        # Use the embedding matrix directly so it is stored and optimized only once.
        logits = self.token_embedding.as_linear(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        return logits, loss

    def generate(self, input_ids, max_new_tokens: int, temperature: float = 1.0):
        if temperature <= 0 or max_new_tokens < 0:
            raise ValueError("temperature must be positive and max_new_tokens nonnegative")
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -self.config.context_length:])
            next_id = mx.random.categorical(logits[:, -1] / temperature)[:, None]
            input_ids = mx.concatenate((input_ids, next_id), axis=1)
            mx.eval(input_ids)  # Bound the lazy graph during long continuations.
        return input_ids
