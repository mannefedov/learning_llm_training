"""A single-device, full-precision next-token training loop.

Training reference: https://github.com/karpathy/build-nanogpt
"""

from dataclasses import dataclass
from collections.abc import Callable

import torch

from llm_training.models.gpt import GPT
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


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: GPT, tokens: torch.Tensor, config: TrainConfig) -> float:
    """Use the same held-out windows each time, without changing training RNG."""
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 1)
    device = next(model.parameters()).device
    total = 0.0
    for _ in range(config.eval_batches):
        x, y = get_batch(tokens, config.batch_size, model.config.context_length, device, generator)
        _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / config.eval_batches


def train(
    model: GPT,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    config: TrainConfig,
    record_metrics: Callable[[int, float | None, float], None] | None = None,
):
    """Optimize for a fixed number of updates and return the optimizer for saving."""
    device = choose_device(config.device)
    model.to(device)
    model.train()
    # Decay matrices, but not LayerNorm scales or biases. Tied weights appear once.
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": config.weight_decay},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )
    generator = torch.Generator().manual_seed(config.seed)
    print(f"Device: {device}; parameters: {sum(p.numel() for p in parameters):,}")
    val_loss = evaluate(model, val_tokens, config)
    print(f"step=0 val_loss={val_loss:.4f}")
    if record_metrics is not None:
        record_metrics(0, None, val_loss)
    for step in range(1, config.steps + 1):
        x, y = get_batch(train_tokens, config.batch_size, model.config.context_length, device, generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            val_loss = evaluate(model, val_tokens, config)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
            if record_metrics is not None:
                record_metrics(step, loss.item(), val_loss)
    return optimizer
