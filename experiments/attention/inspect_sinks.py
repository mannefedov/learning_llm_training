"""Read-only attention probes on fixed validation windows from saved MLX GPTs."""

import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np

from llm_training.models.numerics_mlx import stable_softmax
from llm_training.pretraining_mlx.data import get_batch
from llm_training.pretraining_mlx.generate import load_checkpoint


def inspect(directory, tokens, output):
    model, tokenizer, eos_id = load_checkpoint(directory)
    config = model.config
    if any(config.attention_type(i) != "full" for i in range(config.n_layers)):
        raise ValueError("This probe compares full-attention layers")
    length = config.context_length
    generator = random.Random(31415)
    mean_maps = np.zeros((config.n_layers, config.n_heads, length, length))
    first_wins = np.zeros((config.n_layers, config.n_heads))
    first_eos = 0
    # Restrict summary queries so every query can see the first 64 keys.
    query_start = 64
    assert length > query_start
    for _ in range(8):
        ids, _ = get_batch(tokens, 4, length, generator)
        first_eos += int(mx.sum(ids[:, 0] == eos_id).item())
        x = model.token_embedding(ids)
        if config.position_encoding == "learned":
            x = x + model.position_embedding(mx.arange(length))
        for layer, block in enumerate(model.blocks):
            attention = block.attention
            normalized = block.attention_norm(x)
            q, k, v = mx.split(attention.qkv(normalized), 3, axis=-1)
            head_dim = config.d_model // config.n_heads
            q, k, v = [a.reshape(4, length, config.n_heads, head_dim).transpose(0, 2, 1, 3)
                       for a in (q, k, v)]
            if attention.use_rope:
                q = mx.fast.rope(q, head_dim, traditional=True, base=10000., scale=1., offset=0)
                k = mx.fast.rope(k, head_dim, traditional=True, base=10000., scale=1., offset=0)
            scores = (q @ k.swapaxes(-2, -1)) * attention.attention_scale
            positions = mx.arange(length)
            scores = mx.where(positions[:, None] >= positions[None, :], scores, -float("inf"))
            probabilities = (stable_softmax(scores) if config.stable_softmax
                             else mx.softmax(scores, axis=-1))
            array = np.array(probabilities)
            np.testing.assert_allclose(array.sum(-1), 1, atol=2e-6)
            mean_maps[layer] += array.mean(0) / 8
            first_wins[layer] += (array[:, :, query_start:].argmax(-1) == 0).mean((0, 2)) / 8
            attended = (probabilities @ v).transpose(0, 2, 1, 3).reshape(4, length, config.d_model)
            x = x + attention.proj(attended)
            x = x + block.mlp(block.mlp_norm(x))
        hidden = model.final_norm(x)
        logits = (model.lm_head(hidden) * config.mup_base_width / config.d_model
                  if config.parametrization == "mup" else model.token_embedding.as_linear(hidden))
        # Reconstructed attention must reproduce the saved model's normal entry point.
        np.testing.assert_allclose(np.array(logits), np.array(model(ids)[0]), atol=3e-4, rtol=3e-4)
    late = mean_maps[:, :, query_start:, :]
    mass = late[:, :, :, 0].mean(-1)
    other_early = late[:, :, :, 1:16].mean((-1, -2))
    uniform = float(np.mean(1 / np.arange(query_start + 1, length + 1)))
    summary = {
        "run": directory.name, "checkpoint_step": json.loads((directory / "checkpoint.json").read_text())["step"],
        "position_encoding": config.position_encoding,
        "sequences": 32, "query_start": query_start, "context_length": length,
        "first_tokens_equal_eos": first_eos, "uniform_first_mass": uniform,
        "first_mass": mass.tolist(), "other_early_mass": other_early.tolist(),
        "first_argmax_fraction": first_wins.tolist(),
        "mean_first_mass": float(mass.mean()), "mean_other_early_mass": float(other_early.mean()),
    }
    np.savez_compressed(output / f"{directory.name}.npz", attention=mean_maps)
    return summary


def main():
    output = Path("artifacts/analysis/attention-sinks")
    output.mkdir(parents=True, exist_ok=True)
    tokens = np.memmap("artifacts/data/fineweb2-rus-2b-tokens/val.bin", dtype="<u4", mode="r")
    directories = [
        Path("artifacts/pretraining/gpt-mlx-mup-width256-2b-03"),
        Path("artifacts/pretraining/gpt-mlx-mup-width256-rope-150m-01"),
        Path("artifacts/archive/pre-mup/pretraining/gpt-mlx-basic-diagnostics-width256-2b-01"),
    ]
    summaries = []
    for directory in directories:
        result = inspect(directory, tokens, output)
        summaries.append(result)
        print(json.dumps(result), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
