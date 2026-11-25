"""Independent attention reference, cache equivalence, and train/resume checks."""

from dataclasses import asdict
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("MLX is not installed")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from llm_training.models.config import GPTConfig
from llm_training.models.gpt_mlx import GPT
from llm_training.models.mla_mlx import MultiHeadLatentAttention
from llm_training.pretraining_mlx.train import TrainConfig, make_loss_and_grad, make_optimizer


def tiny_config(**kwargs):
    fields = dict(vocab_size=24, context_length=8, d_model=16, n_heads=2,
                  n_layers=2, position_encoding="rope", attention_architecture="mla",
                  mla_q_rank=6, mla_kv_rank=4, mla_qk_head_dim=3,
                  mla_rope_head_dim=4, mla_v_head_dim=5)
    return GPTConfig(**(fields | kwargs))


class MLATests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(31)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch reference requires torch")
    def test_outputs_and_gradients_against_independent_torch_reference(self):
        import torch
        config = tiny_config()
        layer = MultiHeadLatentAttention(config)
        x = mx.random.normal((2, 5, 16))
        params = {name: torch.tensor(np.array(p), requires_grad=True)
                  for name, p in tree_flatten(layer.parameters())}
        tx = torch.tensor(np.array(x), requires_grad=True)

        def linear(x, name):
            return x @ params[name + ".weight"].T

        def norm(x, name):
            return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * params[name + ".weight"]

        def rotate(x):
            position = torch.arange(5, dtype=torch.float32)
            frequency = 10000. ** (-torch.arange(0, 4, 2, dtype=torch.float32) / 4)
            angle = position[:, None] * frequency
            even, odd = x[..., ::2], x[..., 1::2]
            return torch.stack((even * angle.cos() - odd * angle.sin(),
                                even * angle.sin() + odd * angle.cos()), dim=-1).flatten(-2)

        q = linear(norm(linear(tx, "q_down"), "q_norm"), "q_up").reshape(2, 5, 2, 7).transpose(1, 2)
        qc, qr = q.split([3, 4], dim=-1)
        latent, kr = linear(tx, "kv_down").split([4, 4], dim=-1)
        kv = linear(norm(latent, "kv_norm"), "kv_up").reshape(2, 5, 2, 8).transpose(1, 2)
        kc, values = kv.split([3, 5], dim=-1)
        full_q = torch.cat((qc, rotate(qr)), dim=-1)
        full_k = torch.cat((kc, rotate(kr[:, None]).expand(-1, 2, -1, -1)), dim=-1)
        scores = (full_q @ full_k.transpose(-1, -2)) / 7**.5
        scores = scores.masked_fill(torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1), -float("inf"))
        heads = scores.softmax(-1) @ values
        expected = linear(heads.transpose(1, 2).reshape(2, 5, 10), "proj")
        expected.square().mean().backward()
        actual, gradients = nn.value_and_grad(layer, lambda a: mx.mean(layer(a)**2))(x)
        np.testing.assert_allclose(np.array(layer(x)), expected.detach().numpy(), rtol=2e-5, atol=2e-6)
        for name, gradient in tree_flatten(gradients):
            np.testing.assert_allclose(np.array(gradient), params[name].grad.numpy(), rtol=2e-4, atol=2e-6,
                                       err_msg=name)
        input_gradient = mx.grad(lambda a: mx.mean(layer(a)**2))(x)
        np.testing.assert_allclose(np.array(input_gradient), tx.grad.numpy(), rtol=2e-4, atol=2e-6)

    def test_absorbed_outputs_gradients_and_chunked_cache(self):
        layer = MultiHeadLatentAttention(tiny_config())
        x = mx.random.normal((2, 7, 16))
        full = layer(x)
        absorbed, cache = layer.decode(x)
        np.testing.assert_allclose(np.array(full), np.array(absorbed), rtol=2e-5, atol=2e-6)
        reference_grads = None
        for method in (lambda a: layer(a), lambda a: layer.decode(a)[0]):
            _, grads = nn.value_and_grad(layer, lambda a: mx.mean(method(a)**2))(x)
            if reference_grads is None:
                reference_grads = dict(tree_flatten(grads))
            else:
                for name, value in tree_flatten(grads):
                    np.testing.assert_allclose(np.array(value), np.array(reference_grads[name]),
                                               rtol=2e-4, atol=2e-6, err_msg=name)
        for chunks in ([1]*7, [3, 2, 2]):
            cache, parts, start = None, [], 0
            for size in chunks:
                out, cache = layer.decode(x[:, start:start+size], cache)
                parts.append(out)
                start += size
            np.testing.assert_allclose(np.array(mx.concatenate(parts, axis=1)), np.array(full),
                                       rtol=2e-5, atol=2e-6)
            self.assertEqual(cache[0].shape, (2, 7, 4))
            self.assertEqual(cache[1].shape, (2, 1, 7, 4))
        # Perturb future inputs; earlier outputs must be unchanged.
        altered = mx.concatenate((x[:, :3], x[:, 3:] + 7), axis=1)
        np.testing.assert_array_equal(np.array(layer(x)[:, :3]), np.array(layer(altered)[:, :3]))

    def test_compiled_training_checkpoint_and_moe_composition(self):
        for experts in (0, 4):
            config = tiny_config(num_experts=experts, expert_hidden_dim=16)
            model = GPT(config)
            tokens = mx.array([[1, 2, 3, 4, 5, 6]])
            train_config = TrainConfig(learning_rate=.003, weight_decay=0)
            optimizer = make_optimizer(train_config, config)
            loss_and_grad = make_loss_and_grad(model, True)
            before = model(tokens, tokens)[1].item()
            for _ in range(8):
                loss, grads = loss_and_grad(tokens, tokens)
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)
            self.assertLess(model(tokens, tokens)[1].item(), before)
            self.assertEqual(model.generate(tokens, 2).shape, (1, 8))
            with TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.safetensors"
                model.save_weights(str(path))
                restored = GPT(GPTConfig(**asdict(config)))
                restored.load_weights(str(path))
                np.testing.assert_array_equal(np.array(model(tokens)[0]), np.array(restored(tokens)[0]))

    def test_reference_window_and_chunked_latent_cache(self):
        config = tiny_config(attention_types=["sliding_window", "full"], sliding_window=2)
        layer = MultiHeadLatentAttention(config, 0)
        self.assertIsNone(MultiHeadLatentAttention(config, 1).window_size)
        x = mx.random.normal((2, 7, 16))
        qc, qr, latent, kr = layer.project(x)
        kv = layer.kv_up(latent).reshape(2, 7, 2, 8).transpose(0, 2, 1, 3)
        kc, values = mx.split(kv, [3], axis=-1)
        scores = (qc @ kc.swapaxes(-1, -2) + qr @ kr.swapaxes(-1, -2)) / 7**.5
        distance = mx.arange(7)[:, None] - mx.arange(7)[None, :]
        weights = mx.softmax(mx.where((distance >= 0) & (distance < 2), scores, -float("inf")), axis=-1)
        np.testing.assert_allclose(np.array(layer(x)), np.array(layer.output(weights @ values)), atol=2e-6)
        prefix, cache = layer.decode(x[:, :3])
        suffix, _ = layer.decode(x[:, 3:], cache)
        np.testing.assert_allclose(np.array(mx.concatenate((prefix, suffix), axis=1)),
                                   np.array(layer(x)), rtol=2e-5, atol=2e-6)
        changed = mx.concatenate((x[:, :1] + 7, x[:, 1:]), axis=1)
        np.testing.assert_array_equal(np.array(layer(x)[:, 2:]), np.array(layer(changed)[:, 2:]))

    @unittest.skipUnless(mx.metal.is_available(), "Requires Metal")
    def test_fused_expanded_attention_outputs_and_gradients(self):
        window_supported = "window_size" in (mx.fast.scaled_dot_product_attention.__doc__ or "")
        for window in (None, 5):
            if window is not None and not window_supported:
                continue
            for head_dim in (64, 128):
                with self.subTest(window=window, head_dim=head_dim):
                    options = dict(mla_qk_head_dim=head_dim//2, mla_rope_head_dim=head_dim//2,
                                   mla_v_head_dim=head_dim)
                    if window is not None:
                        options.update(attention_types=["sliding_window", "full"], sliding_window=window)
                    reference = MultiHeadLatentAttention(tiny_config(**options))
                    fused = MultiHeadLatentAttention(tiny_config(**options,
                        attention_backend="mlx_fused" if window is None else "mlx_fused_window"))
                    fused.update(reference.parameters())
                    x = mx.random.normal((2, 33, 16))
                    ref_loss, ref_grads = nn.value_and_grad(reference, lambda a: mx.mean(reference(a)**2))(x)
                    loss, grads = nn.value_and_grad(fused, lambda a: mx.mean(fused(a)**2))(x)
                    np.testing.assert_allclose(np.array(fused(x)), np.array(reference(x)), atol=2e-4, rtol=3e-3)
                    for (name, value), (_, expected) in zip(tree_flatten(grads), tree_flatten(ref_grads)):
                        np.testing.assert_allclose(np.array(value), np.array(expected), atol=2e-4, rtol=3e-3,
                                                   err_msg=name)

    @unittest.skipUnless(mx.metal.is_available() and
        "window_size" in (mx.fast.scaled_dot_product_attention.__doc__ or ""),
        "Requires local windowed MLX build")
    def test_compiled_mixed_window_flash_mla_moe(self):
        config = tiny_config(context_length=33, mla_qk_head_dim=32, mla_rope_head_dim=32,
                             mla_v_head_dim=64, attention_backend="mlx_fused_window",
                             attention_types=["sliding_window", "full"], sliding_window=5,
                             num_experts=4, expert_hidden_dim=16)
        model = GPT(config)
        tokens = mx.arange(33)[None] % config.vocab_size
        optimizer = make_optimizer(TrainConfig(), config)
        loss_and_grad = make_loss_and_grad(model, True)
        for _ in range(2):
            loss, grads = loss_and_grad(tokens, tokens)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
            self.assertTrue(mx.isfinite(loss).item())
            self.assertTrue(all(mx.isfinite(g).all().item() for _, g in tree_flatten(grads)))
        self.assertEqual(model.generate(tokens[:, :1], 3, temperature=0).shape, (1, 4))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.safetensors"
            model.save_weights(str(path))
            restored = GPT(config)
            restored.load_weights(str(path))
            np.testing.assert_array_equal(np.array(model(tokens)[0]), np.array(restored(tokens)[0]))

    def test_invalid_architecture_combinations(self):
        for fields in (dict(mla_kv_rank=0), dict(mla_rope_head_dim=3),
                       dict(position_encoding="learned"), dict(attention_sinks=True),
                       dict(attention_types=["full", "sliding_window"]),
                       dict(attention_backend="mlx_fused"),
                       dict(parametrization="mup", mup_base_width=8)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                tiny_config(**fields)


if __name__ == "__main__":
    unittest.main()
