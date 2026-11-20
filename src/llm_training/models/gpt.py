"""A small GPT-style decoder in pure PyTorch.

Architecture reference: https://github.com/karpathy/build-nanogpt
This is an educational model, not a GPT-2 checkpoint-compatible implementation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import GPTConfig
from .numerics import stable_cross_entropy, stable_softmax
from .attention import CausalSelfAttention, build_attention


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_index: int = 0):
        super().__init__()
        if config.attention_architecture == "mla":
            raise ValueError("MLA currently requires the MLX backend")
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = build_attention(config, layer_index)
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
        if config.num_experts:
            raise ValueError("MoE currently requires the MLX backend")
        if config.attention_backend != "reference":
            raise ValueError("Fused MLX attention requires the MLX training entry point")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if config.position_encoding == "learned":
            self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layers)])
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
        x = self.token_embedding(input_ids)
        if self.config.position_encoding == "learned":
            positions = torch.arange(input_ids.size(1), device=input_ids.device)
            x = x + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = (stable_cross_entropy(logits, targets) if self.config.stable_softmax else
                    F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """Sample continuations, cropping the context when it fills up."""
        if temperature <= 0 or max_new_tokens < 0:
            raise ValueError("temperature must be positive and max_new_tokens nonnegative")
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_length:]
            logits, _ = self(context)
            scores = logits[:, -1] / temperature
            probabilities = stable_softmax(scores) if self.config.stable_softmax else F.softmax(scores, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids
