"""Numerical regressions for confident predictions and masked attention."""

import math
import unittest

import torch
from torch.nn import functional as F

from llm_training.models.numerics import stable_softmax, stable_log_softmax, stable_cross_entropy
from llm_training.models.attention import masked_attention


class NumericsTests(unittest.TestCase):
    def test_confident_loss_and_gradients_on_cpu_and_mps(self):
        mass = 2 * math.exp(-18)
        for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
            with self.subTest(device=device):
                logits = torch.tensor([[9., -9., -9.]], device=device, requires_grad=True)
                loss = stable_cross_entropy(logits, torch.tensor([0], device=device))
                grad = torch.autograd.grad(loss, logits)[0].cpu()
                self.assertAlmostEqual(loss.item() / math.log1p(mass), 1., places=5)
                expected = torch.tensor([[-mass/(1+mass), math.exp(-18)/(1+mass), math.exp(-18)/(1+mass)]])
                torch.testing.assert_close(grad, expected, atol=1e-14, rtol=1e-5)
                self.assertLess(abs(grad.sum().item()), 1e-14)
                q = torch.ones(1, 1, 1, 1, device=device)
                v = torch.tensor([1., 0., 0.], device=device).reshape(1, 1, 3, 1)
                probability = masked_attention(q, logits.reshape(1, 1, 3, 1), v,
                                              torch.ones(1, 3, dtype=torch.bool, device=device), stable=True).sum()
                grad = torch.autograd.grad(probability, logits)[0]
                self.assertAlmostEqual(grad[0, 0].item() / (mass/(1+mass)**2), 1., places=5)

    def test_forward_and_gradients_match_float64_reference(self):
        for values, target in (([1., -2., 0.5], 2), ([0., 0., 0.], 1),
                               ([-1000., 0., 1000.], 0), ([10000., 10001., 9999.], 1)):
            with self.subTest(values=values):
                logits = torch.tensor([values], requires_grad=True)
                reference = logits.detach().double().requires_grad_()
                target = torch.tensor([target])
                actual = stable_cross_entropy(logits, target)
                expected = F.cross_entropy(reference, target)
                torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-6)
                actual_grad = torch.autograd.grad(actual, logits)[0]
                expected_grad = torch.autograd.grad(expected, reference)[0]
                torch.testing.assert_close(actual_grad.double(), expected_grad, atol=2e-7, rtol=2e-6)
                torch.testing.assert_close(stable_softmax(logits).double(), F.softmax(reference, -1),
                                           atol=2e-7, rtol=2e-6)

    def test_single_finite_or_effectively_finite_logit_has_finite_gradients(self):
        for values in ([0., -float("inf"), -float("inf")], [0., -1000., -float("inf")], [5.]):
            logits = torch.tensor([values], requires_grad=True)
            probabilities = stable_softmax(logits)
            gradient = torch.autograd.grad(probabilities[0, 0], logits)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(probabilities[0, 0].item(), 1.)
            self.assertEqual(gradient.abs().sum().item(), 0.)

    def test_double_precision_gradcheck(self):
        logits = torch.tensor([[0.4, -0.2, 1.5]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(stable_log_softmax, (logits,)))
        self.assertTrue(torch.autograd.gradcheck(stable_softmax, (logits,)))
        self.assertTrue(torch.autograd.gradcheck(lambda z: stable_cross_entropy(z, torch.tensor([2])), (logits,)))
