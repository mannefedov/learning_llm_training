"""Tiny effective-batch equivalence, token coverage and clipping checks."""

from contextlib import redirect_stdout
import importlib.util
import io
import unittest
from unittest.mock import patch

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("Install the pretraining-mlx extra")

import mlx.core as mx
from mlx.utils import tree_flatten

from llm_training.models.gpt_mlx import GPT, GPTConfig
from llm_training.pretraining_mlx.train import (
    TrainConfig, train, make_loss_and_grad, accumulate_gradients,
)


class AccumulationTests(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self.device)

    def test_accumulated_gradients_match_single_large_batch(self):
        mx.random.seed(4)
        model = GPT(GPTConfig(vocab_size=8, context_length=4, n_layers=1, n_heads=1, d_model=8))
        x = mx.array([[0, 1, 2, 3], [3, 2, 1, 0], [4, 5, 6, 7], [7, 6, 5, 4]])
        y = (x + 1) % 8
        reference_loss, reference = make_loss_and_grad(model)(x, y)
        mx.eval(reference)
        for compiled in (False, True):
            with self.subTest(compiled=compiled):
                loss, actual = accumulate_gradients(make_loss_and_grad(model, compiled), x, y, 1)
                self.assertAlmostEqual(loss, reference_loss.item(), places=6)
                for (name, expected), (_, observed) in zip(tree_flatten(reference), tree_flatten(actual)):
                    self.assertTrue(mx.allclose(expected, observed, atol=1e-6, rtol=1e-4).item(), name)

    def test_sequential_coverage_and_optimizer_token_counts(self):
        model = GPT(GPTConfig(vocab_size=32, context_length=2, n_layers=1, n_heads=1, d_model=8))
        tokens = mx.arange(17)
        original = make_loss_and_grad(model)
        seen_x, seen_y, metrics = [], [], []

        def observed(x, y):
            self.assertEqual(x.shape, (2, 2))
            seen_x.extend(x.reshape(-1).tolist())
            seen_y.extend(y.reshape(-1).tolist())
            return original(x, y)

        config = TrainConfig(steps=3, batch_size=2, gradient_accumulation_steps=2,
                             sampling="sequential", eval_every=3, eval_batches=1, device="cpu")
        with patch("llm_training.pretraining_mlx.train.make_loss_and_grad", return_value=observed), redirect_stdout(io.StringIO()):
            optimizer = train(model, tokens, tokens, config,
                              record_metrics=lambda step, loss, val, timing: metrics.append((step, timing)))
        # Two full effective batches per epoch, then wrap to the first one.
        self.assertEqual(seen_x, list(range(16)) + list(range(8)))
        self.assertEqual(seen_y, list(range(1, 17)) + list(range(1, 9)))
        self.assertEqual(optimizer.state["step"].item(), 3)
        self.assertEqual(metrics[-1][1]["tokens_processed"], 24)

    def test_clip_once_after_averaging_and_reject_invalid_settings(self):
        import mlx.optimizers as optim
        model = GPT(GPTConfig(vocab_size=8, context_length=2, n_layers=1, n_heads=1, d_model=8))
        tokens = mx.arange(7)
        from llm_training.pretraining_mlx.data import sequential_batch
        x, y = sequential_batch(tokens, 2, 2, 0)
        _, expected = make_loss_and_grad(model)(x, y)
        mx.eval(expected)
        original = optim.clip_grad_norm
        clipped_inputs = []

        def clip(gradients, limit):
            clipped_inputs.append(gradients)
            return original(gradients, limit)

        with patch("llm_training.pretraining_mlx.train.optim.clip_grad_norm", side_effect=clip), redirect_stdout(io.StringIO()):
            train(model, tokens, tokens, TrainConfig(steps=1, batch_size=1, gradient_accumulation_steps=2,
                                                    sampling="sequential", eval_batches=1, device="cpu"))
        self.assertEqual(len(clipped_inputs), 1)
        for (name, reference), (_, actual) in zip(tree_flatten(expected), tree_flatten(clipped_inputs[0])):
            self.assertTrue(mx.allclose(reference, actual, atol=1e-6, rtol=1e-4).item(), name)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrainConfig(gradient_accumulation_steps=value)
