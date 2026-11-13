"""PyTorch causal attention variants with shared QKV/output projections.

Window W includes the current token: query i sees max(0, i-W+1)..i.
SWA computes query tiles of W rows against at most 2W-1 key rows: O(TW)
attention scores rather than O(T²). No KV cache or custom kernel is used.
Reference layouts: https://arxiv.org/abs/2503.19786 (section 5.2).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import GPTConfig


def masked_attention(q, k, v, allowed):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    weights = F.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
    return weights @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def attend(self, q, k, v):
        positions = torch.arange(q.shape[2], device=q.device)
        return masked_attention(q, k, v, positions[:, None] >= positions[None, :])

    def forward(self, x):
        batch, length, width = x.shape
        head_size = width // self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.reshape(batch, length, self.n_heads, head_size).transpose(1, 2)
                   for a in (q, k, v)]
        attended = self.attend(q, k, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
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
            query_positions = torch.arange(start, end, device=q.device)
            key_positions = torch.arange(key_start, end, device=q.device)
            distance = query_positions[:, None] - key_positions[None, :]
            allowed = (distance >= 0) & (distance < self.window_size)
            tiles.append(masked_attention(q[:, :, start:end], k[:, :, key_start:end],
                                          v[:, :, key_start:end], allowed))
        return torch.cat(tiles, dim=2)


def build_attention(config: GPTConfig, layer_index: int):
    kind = config.attention_type(layer_index)
    if kind == "full":
        return CausalSelfAttention(config)
    if kind == "sliding_window":
        return SlidingWindowAttention(config)
    raise ValueError(f"Unknown attention type: {kind}")
