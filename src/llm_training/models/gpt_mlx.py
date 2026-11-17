"""Parallel MLX version of gpt.py; the PyTorch implementation stays independent.

MLX reference: https://ml-explore.github.io/mlx/build/html/index.html
Architecture reference: https://github.com/karpathy/build-nanogpt
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .config import GPTConfig
from .numerics_mlx import stable_cross_entropy
from .attention_mlx import CausalSelfAttention, build_attention


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_index: int = 0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.attention = build_attention(config, layer_index)
        self.mlp_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(approx="precise"),  # MLX name for the original tanh approximation.
            nn.Linear(4 * config.d_model, config.d_model),
        )

    def __call__(self, x, *, activation_observer=None):
        normalized = self.attention_norm(x)
        attention = self.attention(normalized)
        if activation_observer is not None:
            activation_observer("attention_norm", normalized)
            activation_observer("attention_output", attention)
        x = x + attention
        if activation_observer is not None:
            activation_observer("attention_residual", x)
        normalized = self.mlp_norm(x)
        mlp = self.mlp(normalized)
        if activation_observer is not None:
            activation_observer("mlp_norm", normalized)
            activation_observer("mlp_output", mlp)
        x = x + mlp
        if activation_observer is not None:
            activation_observer("mlp_residual", x)
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = [Block(config, i) for i in range(config.n_layers)]
        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-5)

        def initialize(path, parameter):
            if parameter.ndim >= 2:
                # Smaller residual projections match gpt.py initialization.
                residual = path.endswith(("attention.proj.weight", "mlp.layers.2.weight"))
                std = 0.02 / math.sqrt(2 * config.n_layers) if residual else 0.02
                return mx.random.normal(parameter.shape) * std
            return mx.zeros_like(parameter) if path.endswith("bias") else mx.ones_like(parameter)

        self.update(tree_map_with_path(initialize, self.parameters()))

    def __call__(self, input_ids, targets=None, *, activation_observer=None):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.context_length:
            raise ValueError("input_ids must have shape (batch, time) within context_length")
        x = self.token_embedding(input_ids) + self.position_embedding(mx.arange(input_ids.shape[1]))
        if activation_observer is not None:
            activation_observer("embedding", x)
        for index, block in enumerate(self.blocks):
            if activation_observer is None:
                x = block(x)
            else:
                x = block(x, activation_observer=lambda name, value: activation_observer(
                    f"blocks.{index}.{name}", value))
        # Use the embedding matrix directly so it is stored and optimized only once.
        x = self.final_norm(x)
        if activation_observer is not None:
            activation_observer("final_norm", x)
        logits = self.token_embedding.as_linear(x)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = (stable_cross_entropy(logits, targets) if self.config.stable_softmax
                    else nn.losses.cross_entropy(logits, targets, reduction="mean"))
        return logits, loss

    def generate(self, input_ids, max_new_tokens: int, temperature: float = 1.0, *, key=None):
        if temperature <= 0 or max_new_tokens < 0:
            raise ValueError("temperature must be positive and max_new_tokens nonnegative")
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -self.config.context_length:])
            sample_key = None
            if key is not None:
                key, sample_key = mx.random.split(key)
            next_id = mx.random.categorical(logits[:, -1] / temperature, key=sample_key)[:, None]
            input_ids = mx.concatenate((input_ids, next_id), axis=1)
            mx.eval(input_ids)  # Bound the lazy graph during long continuations.
        return input_ids
