"""Full-precision MLX next-token training, parallel to pretraining/train.py.

Optimizer reference: https://ml-explore.github.io/mlx/build/html/python/optimizers.html
"""

from dataclasses import dataclass
from collections.abc import Callable
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from llm_training.models.gpt_mlx import GPT
from .data import get_batch


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 10
    seed: int = 42
    device: str = "auto"

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.eval_every, self.eval_batches) < 1:
            raise ValueError("Training steps and batch counts must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid learning rate, gradient clip, or weight decay")


def choose_device(name: str):
    if name == "auto":
        return mx.gpu if mx.metal.is_available() else mx.cpu
    if name not in ("cpu", "gpu"):
        raise ValueError("MLX device must be auto, cpu, or gpu")
    if name == "gpu" and not mx.metal.is_available():
        raise ValueError("MLX Metal GPU is unavailable")
    return mx.gpu if name == "gpu" else mx.cpu


class MatrixAdamW(optim.AdamW):
    """Match PyTorch's matrix-only decay and bias-corrected AdamW updates."""
    def apply_single(self, gradient, parameter, state):
        if parameter.ndim < 2:
            # Adam performs the same update without the decoupled decay term.
            return optim.Adam.apply_single(self, gradient, parameter, state)
        return super().apply_single(gradient, parameter, state)


def evaluate(model: GPT, tokens, config: TrainConfig) -> float:
    was_training = model.training
    model.eval()
    generator = random.Random(config.seed + 1)
    try:
        total = 0.0
        for _ in range(config.eval_batches):
            x, y = get_batch(tokens, config.batch_size, model.config.context_length, generator)
            total += model(x, y)[1].item()
        return total / config.eval_batches
    finally:
        model.train(was_training)


def train(model: GPT, train_tokens, val_tokens, config: TrainConfig,
          record_metrics: Callable[[int, float | None, float], None] | None = None):
    mx.set_default_device(choose_device(config.device))
    model.train()
    optimizer = MatrixAdamW(learning_rate=config.learning_rate, betas=[0.9, 0.95],
                            weight_decay=config.weight_decay, bias_correction=True)
    loss_and_grad = nn.value_and_grad(model, lambda model, x, y: model(x, y)[1])
    generator = random.Random(config.seed)
    count = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Device: {mx.default_device()}; parameters: {count:,}")
    val_loss = evaluate(model, val_tokens, config)
    print(f"step=0 val_loss={val_loss:.4f}")
    if record_metrics is not None:
        record_metrics(0, None, val_loss)
    for step in range(1, config.steps + 1):
        x, y = get_batch(train_tokens, config.batch_size, model.config.context_length, generator)
        loss, gradients = loss_and_grad(model, x, y)
        if not mx.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at step {step}")
        gradients, _ = optim.clip_grad_norm(gradients, config.grad_clip)
        optimizer.update(model, gradients)
        # MLX is lazy: materialize updates each step to bound the graph and memory.
        mx.eval(model.parameters(), optimizer.state)
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            val_loss = evaluate(model, val_tokens, config)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
            if record_metrics is not None:
                record_metrics(step, loss.item(), val_loss)
    return optimizer
