"""Full-precision MLX next-token training, parallel to pretraining/train.py.

Optimizer reference: https://ml-explore.github.io/mlx/build/html/python/optimizers.html
"""

from dataclasses import dataclass
from collections.abc import Callable
import random
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from llm_training.models.gpt_mlx import GPT
from .data import get_batch, sequential_batch, batches_per_epoch
from llm_training.pretraining.progress import TrainingProgress


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 10
    seed: int = 42
    device: str = "auto"
    sampling: str = "random"
    compile_step: bool = False
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    min_learning_rate: float = 3e-5
    lr_decay_steps: int | None = None

    def __post_init__(self):
        if (not isinstance(self.gradient_accumulation_steps, int)
                or isinstance(self.gradient_accumulation_steps, bool)
                or self.gradient_accumulation_steps < 1):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if not isinstance(self.compile_step, bool):
            raise ValueError("compile_step must be a boolean")
        if self.sampling not in ("random", "sequential"):
            raise ValueError("sampling must be random or sequential")
        if min(self.steps, self.batch_size, self.eval_every, self.eval_batches) < 1:
            raise ValueError("Training steps and batch counts must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid learning rate, gradient clip, or weight decay")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be constant or cosine")
        if (not isinstance(self.warmup_steps, int) or isinstance(self.warmup_steps, bool)
                or self.warmup_steps < 0):
            raise ValueError("warmup_steps must be a nonnegative integer")
        if self.lr_decay_steps is not None and (
                not isinstance(self.lr_decay_steps, int) or isinstance(self.lr_decay_steps, bool)
                or self.lr_decay_steps < 2):
            raise ValueError("lr_decay_steps must be an integer >= 2")
        if self.lr_schedule == "constant":
            if self.warmup_steps or self.lr_decay_steps is not None:
                raise ValueError("Warmup and decay steps require lr_schedule = cosine")
        else:
            end = self.lr_decay_steps if self.lr_decay_steps is not None else self.steps
            if end <= max(1, self.warmup_steps):
                raise ValueError("Cosine decay must end after warmup and the first update")
            if not (math.isfinite(self.learning_rate)
                    and 0 <= self.min_learning_rate <= self.learning_rate):
                raise ValueError("Require finite 0 <= min_learning_rate <= learning_rate")

    @property
    def effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation_steps


def learning_rate_at_step(config: TrainConfig, step: int) -> float:
    """LR for a one-based optimizer update; resume uses the absolute saved step.

    Linear warmup reaches the peak on warmup_steps. Cosine then reaches its
    floor on lr_decay_steps (default: training.steps) and stays there.
    Reference: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
    """
    if config.lr_schedule == "constant":
        return config.learning_rate
    if step <= config.warmup_steps and config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    start = max(1, config.warmup_steps)
    end = config.lr_decay_steps if config.lr_decay_steps is not None else config.steps
    fraction = min(1.0, max(0.0, (step - start) / (end - start)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


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


def make_optimizer(config: TrainConfig):
    return MatrixAdamW(learning_rate=config.learning_rate, betas=[0.9, 0.95],
                       weight_decay=config.weight_decay, bias_correction=True)


def make_loss_and_grad(model: GPT, compile_step: bool = False):
    function = nn.value_and_grad(model, lambda x, y: model(x, y)[1])
    if compile_step:
        # Include both forward and backward. Explicit state inputs/outputs keep
        # current weights visible across optimizer updates and avoid stale captures.
        function = mx.compile(function, inputs=model.state, outputs=model.state)
    return function


def accumulate_gradients(loss_and_grad, x, y, micro_batch_size):
    """Average equal-sized microbatch gradients at fixed weights; clip only afterwards.

    Only token IDs for the effective batch are resident together. Materializing
    the running gradient sum after each backward bounds MLX's lazy graph.
    """
    count = x.shape[0] // micro_batch_size
    if count < 1 or x.shape[0] % micro_batch_size:
        raise ValueError("Effective batch must contain whole microbatches")
    total_loss, gradients = 0.0, None
    for start in range(0, x.shape[0], micro_batch_size):
        loss, current = loss_and_grad(x[start:start + micro_batch_size],
                                      y[start:start + micro_batch_size])
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError("Non-finite microbatch loss")
        total_loss += value
        gradients = current if gradients is None else tree_map(lambda a, b: a + b, gradients, current)
        mx.eval(gradients)
    if count > 1:
        gradients = tree_map(lambda g: g / count, gradients)
    return total_loss / count, gradients


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
          record_metrics: Callable[..., None] | None = None,
          record_samples: Callable[[int], None] | None = None, *,
          optimizer=None, start_step: int = 0, batch_rng_state=None,
          record_checkpoint=None, statistics=None):
    epoch_steps = (batches_per_epoch(train_tokens, config.effective_batch_size, model.config.context_length)
                   if config.sampling == "sequential" else None)
    mx.set_default_device(choose_device(config.device))
    model.train()
    if not 0 <= start_step <= config.steps:
        raise ValueError("start_step must be within the training budget")
    optimizer = make_optimizer(config) if optimizer is None else optimizer
    loss_and_grad = make_loss_and_grad(model, config.compile_step)
    generator = random.Random(config.seed)
    if batch_rng_state is not None:
        generator.setstate(batch_rng_state)
    count = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Device: {mx.default_device()}; parameters: {count:,}")
    with TrainingProgress(config.steps, start_step=start_step, steps_per_epoch=epoch_steps,
                          tokens_per_step=config.effective_batch_size * model.config.context_length) as progress:
        val_loss = evaluate(model, val_tokens, config)
        progress.update(start_step, val_loss=val_loss)
        if record_metrics is not None:
            record_metrics(start_step, None, val_loss,
                           {**progress.metrics(), "learning_rate": None})
        if record_samples is not None:
            record_samples(start_step)
        if statistics is not None:
            statistics.record(start_step)
        for step in range(start_step + 1, config.steps + 1):
            if config.sampling == "sequential":
                x, y = sequential_batch(train_tokens, config.effective_batch_size, model.config.context_length, step - 1)
            else:
                x, y = get_batch(train_tokens, config.effective_batch_size, model.config.context_length, generator)
            train_loss, gradients = accumulate_gradients(loss_and_grad, x, y, config.batch_size)
            snapshot = (statistics.capture(gradients)
                        if statistics is not None and statistics.due(step) else None)
            gradients, gradient_norm = optim.clip_grad_norm(gradients, config.grad_clip)
            learning_rate = learning_rate_at_step(config, step)
            optimizer.learning_rate = learning_rate
            optimizer.update(model, gradients)
            # MLX is lazy: materialize updates each step to bound the graph and memory.
            mx.eval(model.parameters(), optimizer.state)
            if statistics is not None:
                statistics.observe(gradient_norm)
                if snapshot is not None:
                    statistics.record(step, learning_rate, snapshot)
            if record_checkpoint is not None:
                record_checkpoint(step, optimizer, generator)
            if step == 1 or step % config.eval_every == 0 or step == config.steps:
                val_loss = evaluate(model, val_tokens, config)
                progress.update(step, train_loss=train_loss, val_loss=val_loss,
                                learning_rate=learning_rate)
                if record_metrics is not None:
                    record_metrics(step, train_loss, val_loss,
                                   {**progress.metrics(), "learning_rate": learning_rate})
            if record_samples is not None:
                record_samples(step)
            progress.update(step, train_loss=train_loss, learning_rate=learning_rate)
    return optimizer
