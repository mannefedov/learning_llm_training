"""MLX counterpart of numerics.py; same corrected log1p formulation."""

import mlx.core as mx


def stable_log_softmax(logits):
    """Normalize the last axis; each row must contain at least one finite logit."""
    # The choice of pivot is discrete; the gathered pivot value still needs gradients.
    pivot_index = mx.stop_gradient(mx.argmax(logits, axis=-1, keepdims=True))
    pivot = mx.take_along_axis(logits, pivot_index, axis=-1)
    is_pivot = mx.arange(logits.shape[-1]) == pivot_index
    differences = mx.where(is_pivot, 0.0, logits - pivot)
    other_mass = mx.sum(mx.where(is_pivot, 0.0, mx.exp(differences)), axis=-1, keepdims=True)
    return differences - mx.log1p(other_mass)


def stable_softmax(logits):
    return mx.exp(stable_log_softmax(logits))


def stable_cross_entropy(logits, targets):
    log_probabilities = stable_log_softmax(logits)
    return -mx.mean(mx.take_along_axis(log_probabilities, targets[..., None], axis=-1))

