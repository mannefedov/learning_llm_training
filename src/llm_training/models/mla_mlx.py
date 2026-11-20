"""Multi-head latent attention: expanded training and compressed-cache decoding.

Sources: https://arxiv.org/abs/2405.04434 (sections 2.1 and C),
https://arxiv.org/abs/2412.19437 (section 2.1.1), and DeepSeek-V3's
inference/model.py (latent RMSNorm and absorbed attention contractions).
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import GPTConfig
from .numerics_mlx import stable_softmax


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.kv_rank = config.mla_kv_rank
        self.content_dim = config.mla_qk_head_dim
        self.rope_dim = config.mla_rope_head_dim
        self.value_dim = config.mla_v_head_dim
        self.stable_softmax = config.stable_softmax
        self.scale = 1 / math.sqrt(self.content_dim + self.rope_dim)
        self.q_down = nn.Linear(config.d_model, config.mla_q_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.mla_q_rank, eps=1e-6)
        self.q_up = nn.Linear(config.mla_q_rank,
                              self.n_heads * (self.content_dim + self.rope_dim), bias=False)
        # One projection packs the KV latent and the separate, shared RoPE key.
        self.kv_down = nn.Linear(config.d_model, self.kv_rank + self.rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_rank, eps=1e-6)
        self.kv_up = nn.Linear(self.kv_rank,
                               self.n_heads * (self.content_dim + self.value_dim), bias=False)
        self.proj = nn.Linear(self.n_heads * self.value_dim, config.d_model, bias=False)

    def project(self, x, offset=0):
        """Return q_content, rotated q_rope, normalized KV latent, rotated k_rope."""
        batch, length, _ = x.shape
        q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.reshape(batch, length, self.n_heads, -1).transpose(0, 2, 1, 3)
        q_content, q_rope = mx.split(q, [self.content_dim], axis=-1)
        latent, k_rope = mx.split(self.kv_down(x), [self.kv_rank], axis=-1)
        latent = self.kv_norm(latent)
        k_rope = k_rope[:, None, :, :]
        q_rope = mx.fast.rope(q_rope, self.rope_dim, traditional=True,
                              base=10000.0, scale=1.0, offset=offset)
        k_rope = mx.fast.rope(k_rope, self.rope_dim, traditional=True,
                              base=10000.0, scale=1.0, offset=offset)
        return q_content, q_rope, latent, k_rope

    def probabilities(self, scores, offset=0):
        query_positions = offset + mx.arange(scores.shape[-2])
        key_positions = mx.arange(scores.shape[-1])
        scores = mx.where(query_positions[:, None] >= key_positions[None, :],
                          scores * self.scale, -float("inf"))
        return stable_softmax(scores) if self.stable_softmax else mx.softmax(scores, axis=-1)

    def output(self, heads):
        batch, _, length, _ = heads.shape
        return self.proj(heads.transpose(0, 2, 1, 3).reshape(batch, length, -1))

    def __call__(self, x):
        """Parallel training path; explicitly expand content keys and values."""
        q_content, q_rope, latent, k_rope = self.project(x)
        batch, length, _ = latent.shape
        kv = self.kv_up(latent).reshape(batch, length, self.n_heads, -1).transpose(0, 2, 1, 3)
        k_content, values = mx.split(kv, [self.content_dim], axis=-1)
        scores = q_content @ k_content.swapaxes(-2, -1) + q_rope @ k_rope.swapaxes(-2, -1)
        return self.output(self.probabilities(scores) @ values)

    def decode(self, x, cache=None):
        """Append a chunk using only latent [B,T,r] and RoPE key [B,1,T,dr].

        Cache belongs to the caller, contains activations only, and is valid
        while model weights remain fixed. offset is inferred from cached length.
        This reference implementation concatenates chunks; it has no fused kernel.
        """
        if cache is not None and len(cache) != 2:
            raise ValueError("MLA cache must contain latent and RoPE-key arrays")
        offset = 0 if cache is None else cache[0].shape[1]
        if cache is not None and (
                cache[0].shape != (x.shape[0], offset, self.kv_rank)
                or cache[1].shape != (x.shape[0], 1, offset, self.rope_dim)):
            raise ValueError("MLA cache shapes must match batch, KV rank and RoPE dimension")
        q_content, q_rope, latent, k_rope = self.project(x, offset)
        if cache is not None:
            latent = mx.concatenate((cache[0], latent), axis=1)
            k_rope = mx.concatenate((cache[1], k_rope), axis=2)
        # Per-head slices of the packed up-projection: [H, dc+dv, r].
        weights = self.kv_up.weight.reshape(self.n_heads, -1, self.kv_rank)
        key_weights, value_weights = mx.split(weights, [self.content_dim], axis=1)
        # q @ (c @ Wk.T).T == (q @ Wk) @ c.T, with one Wk per head.
        latent_queries = mx.einsum("bhtd,hdr->bhtr", q_content, key_weights)
        scores = latent_queries @ latent[:, None].swapaxes(-2, -1)
        scores = scores + q_rope @ k_rope.swapaxes(-2, -1)
        weights_over_tokens = self.probabilities(scores, offset)
        mixed_latent = weights_over_tokens @ latent[:, None]
        # Sum values in latent space, then apply each head's value up-projection.
        heads = mx.einsum("bhtr,hvr->bhtv", mixed_latent, value_weights)
        return self.output(heads), (latent, k_rope)
