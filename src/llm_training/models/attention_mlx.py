"""MLX attention variants; same window convention and tiling as attention.py."""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import GPTConfig
from .numerics_mlx import stable_softmax


def masked_attention(q, k, v, allowed, *, stable=False):
    scores = (q @ k.swapaxes(-2, -1)) / math.sqrt(q.shape[-1])
    scores = mx.where(allowed, scores, -float("inf"))
    weights = stable_softmax(scores) if stable else mx.softmax(scores, axis=-1)
    return weights @ v


def masked_attention_scaled(q, k, v, allowed, *, scale, stable=False, sink_logits=None):
    scores = (q @ k.swapaxes(-2, -1)) * scale
    scores = mx.where(allowed, scores, -float("inf"))
    if sink_logits is not None:
        # A per-head softmax entry with no value contribution. It is always
        # visible, including for the first token and at sliding-window edges.
        sink = mx.broadcast_to(sink_logits[None, :, None, None], (*scores.shape[:-1], 1))
        scores = mx.concatenate((scores, sink), axis=-1)
    weights = stable_softmax(scores) if stable else mx.softmax(scores, axis=-1)
    if sink_logits is not None:
        weights = weights[..., :-1]
    return weights @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.backend = config.attention_backend
        self.stable_softmax = config.stable_softmax
        self.use_rope = config.position_encoding == "rope"
        if config.attention_sinks:
            self.sink_logits = mx.zeros((config.n_heads,))
        self.attention_scale = (1 / (config.d_model // config.n_heads)
                                if config.parametrization == "mup"
                                else 1 / math.sqrt(config.d_model // config.n_heads))
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def attend(self, q, k, v):
        if self.backend in ("mlx_fused", "mlx_fused_window"):
            # The string mask enables fused causal backward; an array mask does not.
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.attention_scale,
                mask="causal", force_fused=True)
        positions = mx.arange(q.shape[2])
        return masked_attention_scaled(q, k, v, positions[:, None] >= positions[None, :],
                                       scale=self.attention_scale,
                                       stable=self.stable_softmax,
                                       sink_logits=getattr(self, "sink_logits", None))

    def __call__(self, x):
        batch, length, width = x.shape
        head_size = width // self.n_heads
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = [a.reshape(batch, length, self.n_heads, head_size).transpose(0, 2, 1, 3)
                   for a in (q, k, v)]
        if self.use_rope:
            # Use the same adjacent channel pairs as the PyTorch implementation.
            q = mx.fast.rope(q, head_size, traditional=True, base=10000.0, scale=1.0, offset=0)
            k = mx.fast.rope(k, head_size, traditional=True, base=10000.0, scale=1.0, offset=0)
        attended = self.attend(q, k, v)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.proj(attended)


class SlidingWindowAttention(CausalSelfAttention):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        if config.sliding_window is None:
            raise ValueError("sliding_window is required")
        self.window_size = config.sliding_window
        if self.backend == "mlx_fused_window" and "window_size" not in (
            mx.fast.scaled_dot_product_attention.__doc__ or ""
        ):
            raise ValueError("mlx_fused_window requires the local patched MLX environment; "
                             "use .local/venv-mlx-swa/bin/python")

    def attend(self, q, k, v):
        if self.backend == "mlx_fused_window":
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.attention_scale, mask="causal",
                window_size=self.window_size, force_fused=True)
        length = q.shape[2]
        tiles = []
        for start in range(0, length, self.window_size):
            end = min(start + self.window_size, length)
            key_start = max(0, start - self.window_size + 1)
            distance = mx.arange(start, end)[:, None] - mx.arange(key_start, end)[None, :]
            allowed = (distance >= 0) & (distance < self.window_size)
            tiles.append(masked_attention_scaled(q[:, :, start:end], k[:, :, key_start:end],
                                                 v[:, :, key_start:end], allowed,
                                                 scale=self.attention_scale,
                                                 stable=self.stable_softmax,
                                                 sink_logits=getattr(self, "sink_logits", None)))
        return mx.concatenate(tiles, axis=2)


def build_attention(config: GPTConfig, layer_index: int):
    if config.attention_architecture == "mla":
        from .mla_mlx import MultiHeadLatentAttention
        return MultiHeadLatentAttention(config, layer_index)
    kind = config.attention_type(layer_index)
    if kind == "full":
        return CausalSelfAttention(config)
    if kind == "sliding_window":
        return SlidingWindowAttention(config)
    raise ValueError(f"Unknown attention type: {kind}")
