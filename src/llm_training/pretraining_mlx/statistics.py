"""Sampled training diagnostics; no full weights or gradient histories on disk.

Estimator notes and references: docs/pretraining/gradient_noise.md.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import random
import time
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .data import get_batch
from .train import accumulate_gradients
from llm_training.models.numerics_mlx import stable_log_softmax


@dataclass
class StatisticsConfig:
    every: int = 0  # Disabled; independent of validation cadence.
    probe_batches: int = 2
    activations: bool = False
    logits: bool = False
    gradient_noise_batches: int = 0  # 0 disables; otherwise at least two.
    gradient_noise_every: int = 5000

    def __post_init__(self):
        for name in ("every", "probe_batches", "gradient_noise_batches", "gradient_noise_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"statistics.{name} must be a nonnegative integer")
        if self.probe_batches < 1:
            raise ValueError("statistics.probe_batches must be positive")
        if self.gradient_noise_batches == 1:
            raise ValueError("statistics.gradient_noise_batches must be 0 or >= 2")
        if self.gradient_noise_every < 1:
            raise ValueError("statistics.gradient_noise_every must be positive")
        if (self.every and self.gradient_noise_batches
                and self.gradient_noise_every % self.every):
            raise ValueError("gradient_noise_every must be a multiple of statistics.every")
        for name in ("activations", "logits"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"statistics.{name} must be boolean")


def gradient_noise_summary(samples, tokens_per_batch):
    """Raw, unclipped gradients at identical weights on K independently drawn batches.

    S = sum(||g_i - mean(g)||²)/(K-1) estimates batch-gradient variance (trace).
    ||mean(g)||² - S/K estimates squared expected-gradient norm; with few batches
    it can be nonpositive, in which case signal ratios are not identifiable.
    Reference: https://arxiv.org/abs/1812.06162
    """
    count = len(samples)
    if count < 2:
        raise ValueError("Gradient noise needs at least two batches")
    mean = {name: sum(g[name] for g in samples) / count for name in samples[0]}

    def squared_norm(gradient):
        return sum(mx.sum(mx.square(value.astype(mx.float32))) for value in gradient.values())

    mean_squared_norm = squared_norm(mean).item()
    variance = sum(squared_norm({name: g[name] - mean[name] for name in mean}).item()
                   for g in samples) / (count - 1)
    signal = mean_squared_norm - variance / count
    norms = [math.sqrt(squared_norm(g).item()) for g in samples]
    nonzero = [(g, n) for g, n in zip(samples, norms) if n > 0]
    cosine = None
    if len(nonzero) >= 2:
        unit_sum = {name: sum(g[name] / norm for g, norm in nonzero) for name in mean}
        n = len(nonzero)
        cosine = max(-1.0, min(1.0, (squared_norm(unit_sum).item() - n) / (n * (n - 1))))
    ratio = variance / signal if signal > 0 else None
    return {"batches": count, "tokens_per_batch": tokens_per_batch,
            "mean_gradient_norm": math.sqrt(mean_squared_norm),
            "batch_gradient_norm_mean": sum(norms) / count,
            "batch_gradient_variance": variance, "signal_squared_estimate": signal,
            "noise_to_signal": ratio,
            "noise_scale_tokens": tokens_per_batch * ratio if ratio is not None else None,
            "mean_pairwise_cosine": cosine}


def distribution_summary(array):
    values = array.astype(mx.float32)
    return {"mean": mx.mean(values), "std": mx.std(values), "rms": rms(values),
            "abs_max": mx.max(mx.abs(values))}


def logit_summary(logits, targets):
    log_probabilities = stable_log_softmax(logits)
    probabilities = mx.exp(log_probabilities)
    # Clip log probabilities only to avoid 0 * -inf when computing entropy.
    entropy = -mx.sum(probabilities * mx.maximum(log_probabilities, -1e30), axis=-1)
    return {**distribution_summary(logits),
            "vocabulary_std_mean": mx.mean(mx.std(logits, axis=-1)),
            "entropy_mean": mx.mean(entropy),
            "max_probability_mean": mx.mean(mx.max(probabilities, axis=-1)),
            "target_probability_mean": mx.mean(mx.take_along_axis(
                probabilities, targets[..., None], axis=-1)),
            "top1_accuracy": mx.mean(mx.argmax(logits, axis=-1) == targets)}


def scalars(values):
    mx.eval(values)
    return {name: value.item() for name, value in values.items()}


def rms(array):
    return mx.sqrt(mx.mean(mx.square(array.astype(mx.float32))))


class TrainingStatistics:
    def __init__(self, file, model, train_tokens, val_tokens, training, settings):
        self.file = file
        self.model = model
        self.training = training
        self.settings = settings
        self.corpora = {"train": train_tokens, "validation": val_tokens}
        self.noise_loss_and_grad = (nn.value_and_grad(model, lambda x, y: model(x, y)[1])
                                   if settings.gradient_noise_batches else None)
        self.reset_interval()

    def reset_interval(self):
        self.updates = 0
        self.norm_sum = 0.0
        self.norm_max = 0.0
        self.clipped = 0

    def due(self, step):
        return step == 1 or step % self.settings.every == 0 or step == self.training.steps

    def capture(self, gradients):
        """Materialize independent preupdate weights and raw gradient summaries.

        MLX is lazy: these must be evaluated before the optimizer replaces weights.
        Only sampled updates incur the extra parameter-sized snapshot.
        """
        # Exclude deferred training backward/compilation work from the diagnostic
        # timer; it is required for the optimizer regardless of collection.
        mx.eval(gradients)
        started = time.perf_counter()
        before = {name: parameter + mx.zeros_like(parameter)
                  for name, parameter in tree_flatten(self.model.trainable_parameters())}
        gradient_rms = {name: rms(value) for name, value in tree_flatten(gradients)}
        mx.eval(before, gradient_rms)
        return before, gradient_rms, time.perf_counter() - started

    def observe(self, norm):
        # The trainer already computes this pre-clipping norm every update.
        value = norm.item()
        self.updates += 1
        self.norm_sum += value
        self.norm_max = max(self.norm_max, value)
        self.clipped += value > self.training.grad_clip

    def probes(self):
        """Same random windows each time; independent of training's batch RNG."""
        was_training = self.model.training
        self.model.eval()
        results = {}
        try:
            for index, (name, tokens) in enumerate(self.corpora.items()):
                generator = random.Random(self.training.seed + 2 + index)
                total = 0.0
                for batch_index in range(self.settings.probe_batches):
                    x, y = get_batch(tokens, self.training.batch_size,
                                     self.model.config.context_length, generator)
                    if name == "train" and batch_index == 0 and (
                            self.settings.activations or self.settings.logits):
                        activations = {}

                        def observe(name, array):
                            activations[name] = distribution_summary(array)

                        logits, loss = self.model(x, y, activation_observer=(
                            observe if self.settings.activations else None))
                        if self.settings.activations:
                            mx.eval(activations)
                            results["activations"] = {key: scalars(value) for key, value in activations.items()}
                        if self.settings.logits:
                            results["logits"] = scalars(logit_summary(logits, y))
                    else:
                        _, loss = self.model(x, y)
                    total += loss.item()
                results[name + "_loss"] = total / self.settings.probe_batches
        finally:
            self.model.train(was_training)
        return results

    def gradient_noise(self):
        """Separate fixed train windows, at postupdate weights, without optimizer updates."""
        started = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        generator = random.Random(self.training.seed + 4)
        samples = []
        try:
            for _ in range(self.settings.gradient_noise_batches):
                x, y = get_batch(self.corpora["train"], self.training.effective_batch_size,
                                 self.model.config.context_length, generator)
                _, gradients = accumulate_gradients(self.noise_loss_and_grad, x, y, self.training.batch_size)
                mx.eval(gradients)
                samples.append(dict(tree_flatten(gradients)))
            result = gradient_noise_summary(
                samples, self.training.effective_batch_size * self.model.config.context_length)
            result["measurement_seconds"] = time.perf_counter() - started
            return result
        finally:
            self.model.train(was_training)

    def record(self, step, learning_rate=None, snapshot=None):
        started = time.perf_counter()
        tensors = {}
        before, gradients, capture_seconds = snapshot if snapshot is not None else ({}, {}, 0.0)
        arrays = {}
        for name, parameter in tree_flatten(self.model.trainable_parameters()):
            tensors[name] = {"shape": list(parameter.shape), "elements": parameter.size}
            old = before.get(name, parameter)
            values = {"weight_rms": rms(old)}
            if snapshot is not None:
                values["gradient_rms"] = gradients[name]
                values["update_rms"] = rms(parameter - old)
            arrays[name] = values
        mx.eval(arrays)
        for name, values in arrays.items():
            values = {key: value.item() for key, value in values.items()}
            if snapshot is not None:
                # Near-zero tensors (e.g. initial biases) need absolute RMS too.
                values["relative_update"] = (values["update_rms"] / values["weight_rms"]
                                             if values["weight_rms"] > 1e-12 else None)
            tensors[name].update(values)
        count = self.updates
        row = {
            "schema_version": 2, "step": step, "learning_rate": learning_rate,
            "interval": {
                "updates": count,
                "gradient_norm_mean": self.norm_sum / count if count else None,
                "gradient_norm_max": self.norm_max if count else None,
                "clipped_fraction": self.clipped / count if count else None,
            },
            "probes": self.probes(), "tensors": tensors,
        }
        if self.settings.gradient_noise_batches and (
                step in (0, 1, self.training.steps) or step % self.settings.gradient_noise_every == 0):
            row["gradient_noise"] = self.gradient_noise()
        row["measurement_seconds"] = capture_seconds + time.perf_counter() - started
        self.file.write(json.dumps(row, allow_nan=False) + "\n")
        self.file.flush()
        self.reset_interval()


@contextmanager
def log_statistics(directory, model, train_tokens, val_tokens, training, settings):
    if not settings.every:
        yield None
        return
    with (directory / "statistics.jsonl").open("x", encoding="utf-8") as file:
        yield TrainingStatistics(file, model, train_tokens, val_tokens, training, settings)
