"""Two real-data optimizer updates to measure large-model local feasibility."""

import json
from pathlib import Path
from time import perf_counter
import tomllib

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from llm_training.models.config import GPTConfig
from llm_training.models.gpt_mlx import GPT
from llm_training.pretraining_mlx.data import sequential_batch
from llm_training.pretraining_mlx.train import (
    TrainConfig, accumulate_gradients, learning_rate_at_step, make_loss_and_grad, make_optimizer,
)


def main():
    path = Path("configs/pretraining/sequential_gpt_mlx_mup_gpt2_dims_150m.toml")
    config = tomllib.loads(path.read_text())
    mx.set_default_device(mx.gpu)
    mx.random.seed(42)
    model = GPT(GPTConfig(vocab_size=8000, **config["model"]))
    training = TrainConfig(**config["training"])
    optimizer = make_optimizer(training, model.config)
    count = sum(p.size for _, p in tree_flatten(model.parameters()))
    tokens = np.memmap(Path(config["cache"]["directory"]) / "train.bin", dtype="<u4", mode="r")
    mx.eval(model.parameters())
    function = make_loss_and_grad(model, training.compile_step)
    timings = []
    for index in range(2):
        start = perf_counter()
        x, y = sequential_batch(tokens, training.effective_batch_size, model.config.context_length, index)
        loss, grads = accumulate_gradients(function, x, y, training.batch_size)
        grads, norm = optim.clip_grad_norm(grads, training.grad_clip)
        optimizer.learning_rate = learning_rate_at_step(training, index + 1)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        elapsed = perf_counter() - start
        timings.append(elapsed)
        print(json.dumps({"update": index + 1, "loss": loss, "seconds": elapsed,
                          "tok_per_second": 20480 / elapsed}), flush=True)
    report = {
        "config": str(path), "parameters": count, "update_seconds": timings,
        "peak_mlx_memory_gib": mx.get_peak_memory() / 2**30,
        "active_mlx_memory_gib": mx.get_active_memory() / 2**30,
        "notes": "First update includes compilation. Another trainer was active. Two updates test feasibility, not convergence.",
    }
    output = Path("artifacts/benchmarks/gpt2-dims-feasibility.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
