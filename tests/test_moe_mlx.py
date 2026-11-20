"""Selected-expert computation, router gradients, and the production loss path."""

from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import unittest

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("MLX is not installed")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from llm_training.models.config import GPTConfig
from llm_training.models.gpt_mlx import GPT
from llm_training.models.moe_mlx import SparseMoE, route_tokens
from llm_training.pretraining_mlx.train import make_loss_and_grad, make_optimizer, TrainConfig


class MoETests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(42)

    def test_selected_experts_match_dense_reference_and_gradients(self):
        config = GPTConfig(vocab_size=16, d_model=8, num_experts=4,
                           experts_per_token=2, expert_hidden_dim=12)
        model = SparseMoE(config)
        x = mx.random.normal((2, 3, 8))

        def reference(x):
            flat = x.reshape(-1, 8)
            indices, gates, _ = route_tokens(model.router(flat), 2)
            mask = indices[..., None] == mx.arange(4)
            weights = mx.sum(mask * gates[..., None], axis=1)
            outputs = []
            for expert in range(4):
                hidden = model.activation(flat @ model.up.weight[expert].T)
                outputs.append(hidden @ model.down.weight[expert].T)
            return mx.sum(mx.stack(outputs, axis=1) * weights[..., None], axis=1).reshape(x.shape)

        actual, grads = nn.value_and_grad(model, lambda x: mx.sum(model(x)[0] ** 2))(x)
        expected, reference_grads = nn.value_and_grad(model, lambda x: mx.sum(reference(x) ** 2))(x)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)
        for (_, gradient), (_, expected_gradient) in zip(tree_flatten(grads), tree_flatten(reference_grads)):
            np.testing.assert_allclose(np.array(gradient), np.array(expected_gradient), rtol=1e-4, atol=2e-6)

    def test_router_balance_and_top_one_task_gradient(self):
        logits = mx.array([[3., 1., -1., -2.], [2., 1., 0., -1.]])
        indices, gates, stats = route_tokens(logits, 2)
        self.assertEqual(stats["expert_token_counts"].sum().item(), 4)
        np.testing.assert_allclose(np.array(gates.sum(-1)), 1, atol=1e-6)
        balanced = route_tokens(mx.zeros((8, 4)), 2)[2]["balance_loss"]
        self.assertAlmostEqual(balanced.item(), 1.0)
        gradient = mx.grad(lambda scores: route_tokens(scores, 1)[1].sum())(logits)
        self.assertGreater(mx.abs(gradient).sum().item(), 0)

    def test_unselected_experts_get_no_task_gradient(self):
        model = SparseMoE(GPTConfig(vocab_size=8, d_model=4, num_experts=4,
                                   experts_per_token=2, expert_hidden_dim=8))
        model.router.weight = mx.array([[3., 0, 0, 0], [2., 0, 0, 0],
                                        [-2., 0, 0, 0], [-3., 0, 0, 0]])
        x = mx.ones((3, 4))
        _, gradients = nn.value_and_grad(model, lambda x: mx.sum(model(x)[0] ** 2))(x)
        self.assertEqual(mx.abs(gradients["up"]["weight"][2:]).sum().item(), 0)
        self.assertEqual(mx.abs(gradients["down"]["weight"][2:]).sum().item(), 0)
        self.assertGreater(mx.abs(gradients["router"]["weight"]).sum().item(), 0)

    def test_compiled_loss_checkpoint_and_generation(self):
        model = GPT(GPTConfig(vocab_size=16, context_length=8, d_model=16, n_heads=2,
                              n_layers=2, position_encoding="rope", num_experts=4,
                              expert_hidden_dim=24, moe_aux_loss_coef=.01))
        tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        diagnostics = {}
        ce = model(tokens, tokens, routing_observer=lambda name, values: diagnostics.update({name: values}))[1]
        objective = model(tokens, tokens, include_aux_loss=True)[1]
        expected_aux = mx.mean(mx.stack([r["balance_loss"] for r in diagnostics.values()]))
        self.assertAlmostEqual((objective - ce).item(), .01 * expected_aux.item(), places=5)
        loss_and_grad = make_loss_and_grad(model, True)
        optimizer = make_optimizer(TrainConfig(), model.config)
        for _ in range(3):
            loss, gradients = loss_and_grad(tokens, tokens)
            optimizer.update(model, gradients)
            mx.eval(loss, model.parameters(), optimizer.state)
            self.assertTrue(mx.isfinite(loss).item())
        self.assertEqual(model.generate(tokens, 2).shape, (1, 10))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            model.save_weights(str(path))
            restored = GPT(model.config)
            restored.load_weights(str(path))
            np.testing.assert_array_equal(np.array(model(tokens)[0]), np.array(restored(tokens)[0]))
