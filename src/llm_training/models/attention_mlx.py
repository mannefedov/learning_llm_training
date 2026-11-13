"""MLX attention variants; same window convention and tiling as attention.py."""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import GPTConfig


def masked_attention(q, k, v, allowed):
    scores = (q @ k.swapaxes(-2, -1)) / math.sqrt(q.shape[-1])
    weights = mx.softmax(mx.where(allowed, scores, -float("inf")), axis=-1)
    return weights @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def attend(self, q, k, v):
        positions = mx.arange(q.shape[2])
        return masked_attention(q, k, v, positions[:, None] >= positions[None, :])

    def __call__(self, x):
        batch, length, width = x.shape
        head_size = width // self.n_heads
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = [a.reshape(batch, length, self.n_heads, head_size).transpose(0, 2, 1, 3)
                   for a in (q, k, v)]
        attended = self.attend(q, k, v)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.proj(attended)


class SlidingWindowAttention(CausalSelfAttention):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        if config.sliding_window is None:
            raise ValueError("sliding_window is required")
        self.window_size = config.sliding_window

    def attend(self, q, k, v):
        length = q.shape[2]
        tiles = []
        for start in range(0, length, self.window_size):
            end = min(start + self.window_size, length)
            key_start = max(0, start - self.window_size + 1)
            distance = mx.arange(start, end)[:, None] - mx.arange(key_start, end)[None, :]
            allowed = (distance >= 0) & (distance < self.window_size)
            tiles.append(masked_attention(q[:, :, start:end], k[:, :, key_start:end],
                                          v[:, :, key_start:end], allowed))
        return mx.concatenate(tiles, axis=2)


def build_attention(config: GPTConfig, layer_index: int):
    kind = config.attention_type(layer_index)
    if kind == "full":
        return CausalSelfAttention(config)
    if kind == "sliding_window":
        return SlidingWindowAttention(config)
    raise ValueError(f"Unknown attention type: {kind}")
