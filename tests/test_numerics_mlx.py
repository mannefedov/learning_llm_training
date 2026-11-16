"""MLX corrected-math checks, including the actual Metal attention path."""

import importlib.util
import math
import unittest

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("Install the pretraining-mlx extra")

import mlx.core as mx
import mlx.nn as nn

from llm_training.models.numerics_mlx import (
    stable_softmax, stable_cross_entropy,
)
from llm_training.models.attention_mlx import masked_attention
from llm_training.models.gpt_mlx import GPT, GPTConfig


class MlxNumericsTests(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()

    def tearDown(self):
        mx.set_default_device(self.device)

    def test_confident_gradients_and_masks_on_cpu_and_metal(self):
        mass = 2 * math.exp(-18)
        for device in [mx.cpu] + ([mx.gpu] if mx.metal.is_available() else []):
            with self.subTest(device=device):
                mx.set_default_device(device)
                logits = mx.array([[9., -9., -9.]])
                loss, gradient = mx.value_and_grad(lambda z: stable_cross_entropy(z, mx.array([0])))(logits)
                self.assertAlmostEqual(loss.item() / math.log1p(mass), 1., places=5)
                self.assertAlmostEqual(gradient[0, 0].item() / (-mass/(1+mass)), 1., places=5)
                self.assertLess(abs(mx.sum(gradient).item()), 1e-14)
                wide = mx.concatenate((mx.array([[27.]]), mx.zeros((1,7999))), axis=-1)
                _, wide_grad = mx.value_and_grad(lambda z: stable_cross_entropy(z,mx.array([0])))(wide)
                expected_mass = 7999 * math.exp(-27)
                # An 8k-way float32 reduction has more rounding error than the 3-class case.
                ratio = wide_grad[0,0].item() / (-expected_mass/(1+expected_mass))
                self.assertLess(abs(ratio - 1.), 1e-5)
                self.assertLess(abs(mx.sum(wide_grad).item()) / expected_mass, 2e-5)
                q, v = mx.ones((1,1,1,1)), mx.array([1.,0.,0.]).reshape(1,1,3,1)
                output = lambda z: mx.sum(masked_attention(q, z.reshape(1,1,3,1), v,
                                                          mx.ones((1,3),dtype=mx.bool_), stable=True))
                gradient = mx.grad(output)(logits)
                self.assertAlmostEqual(gradient[0, 0].item() / (mass/(1+mass)**2), 1., places=5)
                for values in ([0., -float("inf"), -float("inf")], [0., -1000., -float("inf")], [5.]):
                    z = mx.array([values])
                    gradient = mx.grad(lambda x: stable_softmax(x)[0,0])(z)
                    self.assertTrue(mx.all(mx.isfinite(gradient)).item())
                    self.assertEqual(mx.sum(mx.abs(gradient)).item(), 0.)

    def test_compilation_preserves_gradients_and_masked_rows(self):
        from mlx.utils import tree_flatten
        from llm_training.pretraining_mlx.train import make_loss_and_grad, make_optimizer, TrainConfig
        compiled_stable_softmax = mx.compile(stable_softmax)
        compiled_stable_cross_entropy = mx.compile(stable_cross_entropy)
        for device in [mx.cpu] + ([mx.gpu] if mx.metal.is_available() else []):
            mx.set_default_device(device)
            for values, target in (([9.,-9.,-9.],0), ([0.,0.,0.],1),
                                   ([-1000.,0.,1000.],0), ([0.,-float("inf"),-float("inf")],0)):
                z = mx.array([values])
                actual, gradient = mx.value_and_grad(lambda x: compiled_stable_cross_entropy(x,mx.array([target])))(z)
                expected, expected_grad = mx.value_and_grad(lambda x: stable_cross_entropy(x,mx.array([target])))(z)
                self.assertTrue(mx.all(mx.isfinite(gradient)).item())
                self.assertTrue(mx.allclose(actual,expected,atol=1e-14,rtol=1e-5).item())
                self.assertTrue(mx.allclose(gradient,expected_grad,atol=1e-14,rtol=1e-5).item())
                _, softmax_grad = mx.value_and_grad(lambda x: compiled_stable_softmax(x)[0,0])(z)
                ref_grad = mx.grad(lambda x: stable_softmax(x)[0,0])(z)
                self.assertTrue(mx.all(mx.isfinite(softmax_grad)).item())
                self.assertTrue(mx.allclose(softmax_grad,ref_grad,atol=1e-14,rtol=1e-5).item())
            config = GPTConfig(vocab_size=8,context_length=4,n_layers=2,n_heads=1,d_model=8,
                               attention_types=["sliding_window","full"],sliding_window=1,stable_softmax=True)
            reference = GPT(config)
            compiled = GPT(config)
            compiled.update(reference.parameters())
            ids = mx.array([[0,1,2,3]])
            reference_step = make_loss_and_grad(reference)
            compiled_step = make_loss_and_grad(compiled, True)
            optimizers = [make_optimizer(TrainConfig()) for _ in range(2)]
            for _ in range(2):
                expected, expected_grad = reference_step(ids, ids)
                actual, gradient = compiled_step(ids, ids)
                self.assertTrue(mx.allclose(actual,expected,atol=2e-6).item())
                for (_, value), (_, ref) in zip(tree_flatten(gradient),tree_flatten(expected_grad)):
                    self.assertTrue(mx.all(mx.isfinite(value)).item())
                    self.assertTrue(mx.allclose(value,ref,atol=2e-6,rtol=2e-5).item())
                optimizers[0].update(reference, expected_grad)
                optimizers[1].update(compiled, gradient)
                mx.eval(reference.parameters(), compiled.parameters(), [o.state for o in optimizers])

    def test_ordinary_extreme_wrong_targets_and_model_gradients(self):
        for device in [mx.cpu] + ([mx.gpu] if mx.metal.is_available() else []):
            mx.set_default_device(device)
            for values, target in (([1., -2., 0.5], 2), ([0., 0., 0.], 1),
                                   ([-1000., 0., 1000.], 0), ([10000., 10001., 9999.], 1)):
                logits = mx.array([values])
                actual, gradient = mx.value_and_grad(lambda z: stable_cross_entropy(z,mx.array([target])))(logits)
                expected, ref_grad = mx.value_and_grad(lambda z: nn.losses.cross_entropy(
                    z,mx.array([target]),reduction="mean"))(logits)
                self.assertTrue(mx.allclose(actual, expected, atol=2e-6).item())
                self.assertTrue(mx.allclose(gradient, ref_grad, atol=2e-6).item())
            # Window=1 and the first causal row exercise the erratum during model backward.
            model = GPT(GPTConfig(vocab_size=8,context_length=4,n_layers=2,n_heads=1,d_model=8,
                                 attention_types=["sliding_window","full"], sliding_window=1, stable_softmax=True))
            ids = mx.array([[0,1,2,3]])
            loss, grads = nn.value_and_grad(model,lambda m: m(ids,ids)[1])(model)
            from mlx.utils import tree_flatten
            self.assertTrue(mx.isfinite(loss).item())
            self.assertTrue(all(mx.all(mx.isfinite(g)).item() for _,g in tree_flatten(grads)))
