"""Measure eight-update functional changes across widths and initializations."""

import json

import mlx.core as mx
import mlx.nn as nn

from llm_training.models.config import GPTConfig
from llm_training.models.gpt_mlx import GPT
from llm_training.pretraining_mlx.train import TrainConfig, make_optimizer


def rms(values):
    return mx.sqrt(mx.mean(mx.square(values.astype(mx.float32)))).item()


def main():
    tokens = mx.arange(256).reshape(8, 32) % 128
    targets = mx.concatenate([tokens[:, 1:], tokens[:, :1]], axis=1)
    for seed in (7, 17, 27):
      for width in (64, 128, 256):
        mx.random.seed(seed)
        config = GPTConfig(vocab_size=128, context_length=32, n_layers=2,
                           n_heads=4, d_model=width, parametrization="mup",
                           mup_base_width=32)
        model = GPT(config)
        before = {}
        logits_before, loss_before = model(
            tokens, targets, activation_observer=lambda name, value: before.update({name: value}))
        mx.eval(logits_before, loss_before, before)
        optimizer = make_optimizer(TrainConfig(steps=1, batch_size=8,
                                                learning_rate=3e-4, weight_decay=0), config)
        loss_and_grad = nn.value_and_grad(model, lambda x, y: model(x, y)[1])
        for _ in range(8):
            _, gradients = loss_and_grad(tokens, targets)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
        after = {}
        logits_after, loss_after = model(
            tokens, targets, activation_observer=lambda name, value: after.update({name: value}))
        mx.eval(logits_after, loss_after, after)
        delta = logits_after - logits_before
        target_delta = mx.take_along_axis(delta, targets[..., None], axis=-1)[..., 0]
        centered_delta = delta - mx.mean(delta, axis=-1, keepdims=True)
        names = ("embedding", "blocks.0.attention_output", "blocks.0.mlp_output",
                 "blocks.0.mlp_residual", "blocks.1.mlp_residual", "final_norm")
        print(json.dumps({"seed": seed, "width": width, "loss_before": loss_before.item(),
                          "loss_after": loss_after.item(),
                          "loss_improvement": loss_before.item() - loss_after.item(),
                          "logit_rms_before": rms(logits_before),
                          "logit_change_rms": rms(delta),
                          "centered_logit_change_rms": rms(centered_delta),
                          "target_logit_change_mean": mx.mean(target_delta).item(),
                          "activation_rms_before": {name: rms(before[name]) for name in names},
                          "activation_change_rms": {
                              name: rms(after[name] - before[name]) for name in names}}))


if __name__ == "__main__":
    main()
