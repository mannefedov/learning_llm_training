"""PyTorch softmax without losing the gradient of a dominant logit.

Adapted from the corrected log1p formulation (including masked rows) in:
https://jonbarron.info/softmaxfix/ (section 2 and erratum).
This covers attention and hard-target cross-entropy, not distillation losses.
"""

import torch


def stable_log_softmax(logits):
    """Normalize the last axis; each row must contain at least one finite logit."""
    pivot_index = logits.argmax(dim=-1, keepdim=True)
    pivot = logits.gather(-1, pivot_index)
    is_pivot = torch.arange(logits.shape[-1], device=logits.device) == pivot_index
    # Keep the pivot slot constant. Its gradient comes through the differences
    # in all other slots, so it does not require subtracting a probability from 1.
    differences = torch.where(is_pivot, 0.0, logits - pivot)
    other_mass = torch.where(is_pivot, 0.0, differences.exp()).sum(-1, keepdim=True)
    # log1p(0) has a finite derivative, including a causal row with one key.
    # Do not replace this with softplus(log(other_mass)): log(0) breaks backward.
    return differences - torch.log1p(other_mass)


def stable_softmax(logits):
    return stable_log_softmax(logits).exp()


def stable_cross_entropy(logits, targets):
    log_probabilities = stable_log_softmax(logits)
    return -log_probabilities.gather(-1, targets.unsqueeze(-1)).mean()
