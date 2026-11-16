"""Check forced Metal attention against the explicit causal reference."""

import importlib.util
import unittest

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("Install the pretraining-mlx extra")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from llm_training.models.attention_mlx import CausalSelfAttention
from llm_training.models.config import GPTConfig
from llm_training.models.gpt_mlx import GPT


@unittest.skipUnless(mx.metal.is_available(), "Requires Metal")
class FusedAttentionTests(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()
        mx.set_default_device(mx.gpu)
        mx.random.seed(42)

    def tearDown(self):
        mx.set_default_device(self.device)

    def test_outputs_and_qkv_gradients_match_reference(self):
        reference = CausalSelfAttention(GPTConfig(vocab_size=8, d_model=128, n_heads=2))
        fused = CausalSelfAttention(GPTConfig(vocab_size=8, d_model=128, n_heads=2,
                                             attention_backend="mlx_fused"))
        # Metal fused float32 arithmetic differs by ~2e-3 on these inputs.
        for length in (1, 7, 16, 33, 256):
            q, k, v, upstream = [mx.random.normal((1, 2, length, 64)) for _ in range(4)]
            expected = reference.attend(q, k, v)
            actual = fused.attend(q, k, v)
            self.assertTrue(mx.allclose(actual, expected, atol=3e-3, rtol=3e-3).item())
            # Short-prefix generation uses a vector kernel without backward.
            if length <= 8:
                continue
            def loss(a, b, c, attention):
                return mx.sum(attention.attend(a, b, c) * upstream)
            ref_grads = mx.grad(lambda a,b,c: loss(a,b,c,reference), argnums=(0,1,2))(q,k,v)
            fused_grads = mx.grad(lambda a,b,c: loss(a,b,c,fused), argnums=(0,1,2))(q,k,v)
            for actual, expected in zip(fused_grads, ref_grads):
                self.assertTrue(mx.allclose(actual, expected, atol=3e-3, rtol=3e-3).item())

    @unittest.skipUnless(
        "window_size" in (mx.fast.scaled_dot_product_attention.__doc__ or ""),
        "Requires local windowed MLX build")
    def test_windowed_model_matches_reference(self):
        options = dict(vocab_size=16, context_length=33, n_layers=2,
                       d_model=128, n_heads=2, sliding_window=7,
                       attention_types=["sliding_window", "full"])
        reference = GPT(GPTConfig(**options))
        fused = GPT(GPTConfig(**options, attention_backend="mlx_fused_window"))
        fused.update(reference.parameters())
        ids = mx.arange(33)[None, :] % 16
        ref_loss, ref_grads = nn.value_and_grad(reference, lambda x: reference(x,x)[1])(ids)
        fn = mx.compile(nn.value_and_grad(fused, lambda x: fused(x,x)[1]),
                        inputs=fused.state, outputs=fused.state)
        loss, grads = fn(ids)
        self.assertAlmostEqual(loss.item(), ref_loss.item(), places=4)
        for (_, actual), (_, expected) in zip(tree_flatten(grads), tree_flatten(ref_grads)):
            self.assertTrue(mx.allclose(actual, expected, atol=0.001, rtol=0.01).item())
        self.assertEqual(fused.generate(ids[:, :1], 2).shape, (1,3))

    def test_compiled_model_backward_and_generation(self):
        model = GPT(GPTConfig(vocab_size=16, context_length=16, n_layers=1,
                              d_model=128, n_heads=2, attention_backend="mlx_fused"))
        ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        function = mx.compile(nn.value_and_grad(model, lambda x: model(x, x)[1]),
                              inputs=model.state, outputs=model.state)
        loss, grads = function(ids)
        self.assertTrue(mx.isfinite(loss).item())
        for _, gradient in tree_flatten(grads):
            self.assertTrue(mx.all(mx.isfinite(gradient)).item())
        self.assertEqual(model.generate(ids[:, :1], 2).shape, (1, 3))


class FusedConfigTests(unittest.TestCase):
    def test_incompatible_math_and_masks_rejected(self):
        for options in ({"stable_softmax": True},
                        {"attention_types": ["sliding_window"] * 4, "sliding_window": 4}):
            with self.assertRaises(ValueError):
                GPTConfig(vocab_size=8, attention_backend="mlx_fused", **options)
