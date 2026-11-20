"""Small token-choice MoE with selected-expert matrix multiplies.

Top-k selection follows Mixtral (https://arxiv.org/abs/2401.04088).
Balance loss extends Switch's assignment fraction to k choices per token
(https://arxiv.org/abs/2101.03961). Expert choices are discrete; selected
gates and the mean router probabilities supply router gradients.
"""

import mlx.core as mx
import mlx.nn as nn

from .config import GPTConfig


def route_tokens(logits, top_k):
    """Return expert indices and gates for flattened [tokens, experts] scores."""
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    indices = mx.stop_gradient(mx.argpartition(-logits, kth=top_k - 1, axis=-1)[:, :top_k])
    selected = mx.take_along_axis(logits, indices, axis=-1)
    # Normalized top-1 would always equal one, eliminating its task gradient.
    gates = (mx.take_along_axis(probabilities, indices, axis=-1) if top_k == 1
             else mx.softmax(selected.astype(mx.float32), axis=-1))
    assignments = indices[..., None] == mx.arange(logits.shape[-1])
    counts = mx.sum(assignments, axis=(0, 1))
    fraction = mx.stop_gradient(counts / indices.size)
    mean_probability = mx.mean(probabilities, axis=0)
    balance = logits.shape[-1] * mx.sum(fraction * mean_probability)
    entropy = -mx.mean(mx.sum(probabilities * mx.log(mx.maximum(probabilities, 1e-9)), axis=-1))
    return indices, gates, {
        "expert_token_counts": counts,
        "mean_router_probability": mean_probability,
        "balance_loss": balance,
        "router_entropy": entropy,
    }


class ExpertLinear(nn.Module):
    """One weight matrix per expert, applied only to the selected token slots."""

    def __init__(self, num_experts, input_dim, output_dim):
        super().__init__()
        self.weight = mx.random.normal((num_experts, output_dim, input_dim)) * 0.02

    def __call__(self, x, token_indices, expert_indices):
        result = mx.gather_mm(x, self.weight.swapaxes(-1, -2),
                              lhs_indices=token_indices, rhs_indices=expert_indices)
        return result


class SparseMoE(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.top_k = config.experts_per_token
        hidden = config.expert_hidden_dim or 4 * config.d_model
        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.up = ExpertLinear(config.num_experts, config.d_model, hidden)
        self.down = ExpertLinear(config.num_experts, hidden, config.d_model)
        self.activation = nn.GELU(approx="precise")

    def __call__(self, x):
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        indices, gates, diagnostics = route_tokens(self.router(flat), self.top_k)
        # Each token is duplicated into k slots. We gather matrices, with no
        # host-side routing loop, capacity limit, or dropped assignments.
        token_indices = mx.broadcast_to(mx.arange(flat.shape[0])[:, None], indices.shape)
        hidden = self.activation(self.up(flat[:, None, :], token_indices, indices))
        slot_indices = mx.arange(indices.size).reshape(indices.shape)
        expert_output = self.down(hidden.reshape(-1, 1, hidden.shape[-1]), slot_indices, indices)
        mixed = mx.sum(expert_output[..., 0, :] * gates[..., None], axis=1)
        return mixed.reshape(original_shape), diagnostics
