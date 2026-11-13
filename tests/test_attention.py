"""Attention-window boundaries, tiled gradients, and old checkpoint defaults."""

from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from llm_training.models.attention import CausalSelfAttention, SlidingWindowAttention
from llm_training.models.config import GPTConfig
from llm_training.models.gpt import GPT


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_tiled_attention_and_gradients_match_dense_reference(self):
        import llm_training.models.attention as attention
        for window in (1, 3, 11, 20):
            with self.subTest(window=window):
                config = GPTConfig(vocab_size=8, n_layers=1, d_model=8, n_heads=2,
                                   attention_types=["sliding_window"], sliding_window=window)
                model = SlidingWindowAttention(config)
                q, k, v = [torch.randn(2, 2, 11, 4, requires_grad=True) for _ in range(3)]
                with patch.object(attention, "masked_attention", wraps=attention.masked_attention) as kernel:
                    actual = model.attend(q, k, v)
                self.assertTrue(all(call.args[0].shape[2] <= window for call in kernel.call_args_list))
                self.assertTrue(all(call.args[1].shape[2] <= 2 * window - 1 for call in kernel.call_args_list))
                positions = torch.arange(11)
                distance = positions[:, None] - positions[None, :]
                scores = (q @ k.transpose(-2, -1)) / 2
                reference = torch.softmax(scores.masked_fill(
                    ~((distance >= 0) & (distance < window)), -float("inf")), dim=-1) @ v
                torch.testing.assert_close(actual, reference)
                upstream = torch.randn_like(actual)
                actual_grads = torch.autograd.grad((actual * upstream).sum(), (q, k, v), retain_graph=True)
                reference_grads = torch.autograd.grad((reference * upstream).sum(), (q, k, v))
                for actual_grad, reference_grad in zip(actual_grads, reference_grads):
                    torch.testing.assert_close(actual_grad, reference_grad, atol=2e-6, rtol=2e-5)

    def test_window_includes_self_and_excludes_future_and_distant_past(self):
        config = GPTConfig(vocab_size=8, n_layers=1, sliding_window=3)
        local = SlidingWindowAttention(config)
        q = k = torch.zeros(1, 1, 7, 1)
        v = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7, 1)
        result = local.attend(q, k, v).flatten()
        torch.testing.assert_close(result, torch.tensor([0., 0.5, 1., 2., 3., 4., 5.]))
        changed = v.clone()
        changed[:, :, 0] = 1000
        changed[:, :, 6] = 2000
        torch.testing.assert_close(local.attend(q, k, changed)[:, :, 3:6], result[3:6].reshape(1, 1, 3, 1))

    def test_default_full_and_large_window_match_with_same_weights(self):
        config = GPTConfig(vocab_size=8, context_length=7, n_layers=2, n_heads=2, d_model=8)
        old_dimensions = {k: v for k, v in asdict(config).items() if k not in ("attention_types", "sliding_window")}
        full = GPT(GPTConfig(**old_dimensions))
        local_config = GPTConfig(**old_dimensions, attention_types=["sliding_window"] * 2, sliding_window=7)
        local = GPT(local_config)
        local.load_state_dict(full.state_dict())
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        torch.testing.assert_close(full(ids)[0], local(ids)[0])
        restored = GPT(GPTConfig(**asdict(local_config)))
        restored.load_state_dict(local.state_dict())
        torch.testing.assert_close(restored(ids)[0], local(ids)[0])
        self.assertTrue(all(isinstance(block.attention, CausalSelfAttention) for block in full.blocks))

    def test_mixed_layer_selection_and_invalid_config(self):
        config = GPTConfig(vocab_size=8, n_layers=4, attention_types=["sliding_window", "full"] * 2,
                           sliding_window=2)
        model = GPT(config)
        self.assertEqual([type(b.attention) for b in model.blocks],
                         [SlidingWindowAttention, CausalSelfAttention] * 2)
        for kwargs in ({"attention_types": ["full"]}, {"attention_types": ["unknown"] * 4},
                       {"attention_types": ["sliding_window"] * 4}, {"sliding_window": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GPTConfig(vocab_size=8, **kwargs)
