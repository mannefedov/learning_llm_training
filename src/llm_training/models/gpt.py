"""A small GPT-style decoder in pure PyTorch.

Architecture reference: https://github.com/karpathy/build-nanogpt
This is an educational model, not a GPT-2 checkpoint-compatible implementation.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


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
        self.register_buffer(
            "causal_mask",
            torch.ones(config.context_length, config.context_length, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        head_size = width // self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # Each head attends independently: (batch, heads, time, head_size).
        q = q.view(batch, length, self.n_heads, head_size).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, head_size).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, head_size).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(attended)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.d_model, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # The same matrix maps IDs to embeddings and hidden states to logits.
        self.lm_head.weight = self.token_embedding.weight
        # Smaller residual projections keep their combined scale controlled.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.proj.weight, std=residual_std)
            nn.init.normal_(block.mlp[2].weight, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if input_ids.ndim != 2 or not 1 <= input_ids.size(1) <= self.config.context_length:
            raise ValueError("input_ids must have shape (batch, time) within context_length")
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """Sample continuations, cropping the context when it fills up."""
        if temperature <= 0 or max_new_tokens < 0:
            raise ValueError("temperature must be positive and max_new_tokens nonnegative")
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_length:]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1] / temperature, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids
