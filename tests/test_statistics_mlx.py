"""Tiny CPU checks for sampled statistics and observational training behavior."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("Install the pretraining-mlx extra")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from llm_training.models.gpt_mlx import GPT, GPTConfig
from llm_training.pretraining_mlx.statistics import (
    StatisticsConfig, TrainingStatistics, log_statistics, gradient_noise_summary,
    distribution_summary, logit_summary, scalars,
)
from llm_training.pretraining_mlx.train import TrainConfig, train


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self.device)

    def test_raw_gradients_actual_deltas_and_interval_reset(self):
        model = nn.Module()
        model.weight = mx.array([3.0, 4.0])
        model.bias = mx.zeros((2,))
        file = io.StringIO()
        stats = TrainingStatistics(file, model, None, None, TrainConfig(), StatisticsConfig(every=2))
        snapshot = stats.capture({"weight": mx.array([6.0, 8.0]), "bias": mx.ones((2,))})
        model.weight = model.weight * 0.5
        model.bias = mx.ones((2,))
        stats.observe(mx.array(0.5))
        stats.observe(mx.array(2.0))
        with patch.object(stats, "probes", return_value={}):
            stats.record(2, 3e-4, snapshot)
            stats.record(3)
        first, second = [json.loads(line) for line in file.getvalue().splitlines()]
        weight = first["tensors"]["weight"]
        self.assertAlmostEqual(weight["gradient_rms"], (50 ** 0.5), places=5)
        self.assertAlmostEqual(weight["weight_rms"], (12.5 ** 0.5), places=5)
        self.assertAlmostEqual(weight["update_rms"], (12.5 ** 0.5) / 2, places=5)
        self.assertAlmostEqual(weight["relative_update"], 0.5)
        self.assertEqual(first["tensors"]["bias"]["update_rms"], 1.0)
        self.assertEqual(first["interval"], dict(updates=2, gradient_norm_mean=1.25,
                                              gradient_norm_max=2.0, clipped_fraction=0.5))
        self.assertEqual(second["interval"]["updates"], 0)
        self.assertIsNone(second["interval"]["clipped_fraction"])

    def test_statistics_do_not_change_training_or_batch_rng(self):
        for compiled, accumulation in ((False, 1), (True, 1), (True, 2)):
            with self.subTest(compiled=compiled, accumulation=accumulation), redirect_stdout(io.StringIO()):
                mx.random.seed(8)
                config = GPTConfig(vocab_size=4, context_length=4, n_layers=1, n_heads=1, d_model=8)
                baseline = GPT(config)
                mx.eval(baseline.parameters())
                measured = GPT(config)
                measured.load_weights(tree_flatten(baseline.parameters()))
                training = TrainConfig(steps=4, batch_size=2, device="cpu", eval_every=3,
                                       eval_batches=1, compile_step=compiled,
                                       gradient_accumulation_steps=accumulation)
                tokens = mx.array([0, 1, 2, 3] * 10)
                file = io.StringIO()
                stats = TrainingStatistics(file, measured, tokens, tokens, training,
                                           StatisticsConfig(every=3, probe_batches=1, activations=True,
                                                            logits=True, gradient_noise_batches=3,
                                                            gradient_noise_every=3))
                self.assertEqual(stats.probes(), stats.probes())
                self.assertTrue(measured.training)
                measured.eval()
                stats.probes()
                self.assertFalse(measured.training)
                losses = [[], []]
                for index, model in enumerate((baseline, measured)):
                    train(model, tokens, tokens, training,
                          record_metrics=lambda step, loss, val, timing: losses[index].append((step, loss, val)),
                          statistics=stats if index else None)
                self.assertEqual(losses[0], losses[1])
                for (name, expected), (_, actual) in zip(tree_flatten(baseline.parameters()),
                                                       tree_flatten(measured.parameters())):
                    self.assertTrue(mx.array_equal(expected, actual).item(), name)
                rows = [json.loads(line) for line in file.getvalue().splitlines()]
                self.assertEqual([r["step"] for r in rows], [0, 1, 3, 4])
                self.assertEqual([r["interval"]["updates"] for r in rows], [0, 1, 2, 1])
                self.assertNotIn("update_rms", rows[0]["tensors"]["token_embedding.weight"])
                self.assertGreater(rows[1]["tensors"]["token_embedding.weight"]["update_rms"], 0)
                self.assertTrue(all(r["measurement_seconds"] >= 0 for r in rows))
                self.assertTrue(all(r["schema_version"] == 2 for r in rows))
                self.assertTrue(all("gradient_noise" in r for r in rows))
                self.assertEqual(rows[-1]["gradient_noise"]["tokens_per_batch"], 8 * accumulation)
                self.assertIn("blocks.0.attention_output", rows[-1]["probes"]["activations"])
                self.assertGreater(rows[-1]["probes"]["logits"]["entropy_mean"], 0)

    def test_gradient_noise_estimator_on_known_vectors(self):
        samples = [{"w": mx.array(v)} for v in ([1., 2.], [3., 4.], [5., 6.])]
        result = gradient_noise_summary(samples, tokens_per_batch=10)
        self.assertAlmostEqual(result["mean_gradient_norm"], 5)
        self.assertAlmostEqual(result["batch_gradient_variance"], 8)
        self.assertAlmostEqual(result["signal_squared_estimate"], 25 - 8/3)
        self.assertAlmostEqual(result["noise_to_signal"], 8/(25 - 8/3))
        self.assertAlmostEqual(result["noise_scale_tokens"], 80/(25 - 8/3))
        dot = sum(sum(a*b for a, b in zip(x, y)) / (sum(a*a for a in x)*sum(b*b for b in y))**0.5
                  for x, y in [([1, 2], [3, 4]), ([1, 2], [5, 6]), ([3, 4], [5, 6])]) / 3
        self.assertAlmostEqual(result["mean_pairwise_cosine"], dot, places=6)
        for vectors, cosine in [([[1., 0.], [1., 0.]], 1),
                                ([[1., 0.], [-1., 0.]], -1),
                                ([[0., 0.], [0., 0.]], None)]:
            result = gradient_noise_summary([{"w": mx.array(v)} for v in vectors], 10)
            self.assertEqual(result["mean_pairwise_cosine"], cosine)
            if cosine != 1:
                self.assertIsNone(result["noise_scale_tokens"])
            else:
                self.assertEqual(result["batch_gradient_variance"], 0)

    def test_activation_and_logit_summaries_and_observed_forward(self):
        import math
        summary = scalars(distribution_summary(mx.array([-1., 1.])))
        self.assertEqual(summary, {"mean": 0, "std": 1, "rms": 1, "abs_max": 1})
        summary = scalars(logit_summary(mx.zeros((1, 2, 4)), mx.array([[0, 1]])))
        self.assertAlmostEqual(summary["entropy_mean"], math.log(4))
        self.assertEqual(summary["max_probability_mean"], 0.25)
        self.assertEqual(summary["target_probability_mean"], 0.25)
        self.assertEqual(summary["top1_accuracy"], 0.5)
        model = GPT(GPTConfig(vocab_size=4, context_length=4, n_layers=1, n_heads=1, d_model=8))
        x = mx.array([[0, 1, 2, 3]])
        observed = {}
        actual = model(x, activation_observer=lambda key, value: observed.update({key: value}))[0]
        expected = model(x)[0]
        self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertEqual(len(observed), 8)  # Embedding, six block points, final norm.
        stats = TrainingStatistics(io.StringIO(), model, x.reshape(-1), x.reshape(-1),
                                   TrainConfig(), StatisticsConfig(gradient_noise_batches=2))
        model.eval()
        with patch.object(stats, "noise_loss_and_grad", side_effect=RuntimeError("probe failed")):
            # Enough tokens to sample one full context.
            stats.corpora["train"] = mx.array([0, 1, 2, 3, 0, 1])
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                stats.gradient_noise()
        self.assertFalse(model.training)

    def test_disabled_and_invalid_settings(self):
        with TemporaryDirectory() as directory:
            with log_statistics(Path(directory), None, None, None, None, StatisticsConfig()) as stats:
                self.assertIsNone(stats)
            self.assertFalse(list(Path(directory).iterdir()))
        for fields in (dict(every=-1), dict(every=True), dict(every=1.5), dict(probe_batches=0),
                       dict(gradient_noise_batches=1), dict(gradient_noise_batches=-1),
                       dict(gradient_noise_every=0), dict(activations=1), dict(logits="true"),
                       dict(every=3, gradient_noise_batches=2, gradient_noise_every=4)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                StatisticsConfig(**fields)
