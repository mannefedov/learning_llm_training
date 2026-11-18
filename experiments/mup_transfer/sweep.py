"""Short MLX proxy sweep for μP base learning rate.

The target-width learning rate follows the proxy width rule. The report keeps
both values so that the transfer rule can be inspected and compared later.
"""

import argparse
import json
import math
from pathlib import Path
import random
from time import perf_counter
from itertools import product



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True,
                        help="Existing cache directory containing train.bin and val.bin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--proxy-width", type=int, default=128)
    parser.add_argument("--target-width", type=int, default=256)
    parser.add_argument("--base-width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--learning-rates", type=float, nargs="+",
                        default=[0.0012, 0.0024, 0.0048, 0.0096])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    args = parser.parse_args()
    if args.proxy_width % 4 or args.target_width % 4 or args.base_width % 4:
        raise ValueError("base, proxy and target widths must be divisible by four")
    if args.target_width < args.proxy_width:
        raise ValueError("target width must be >= proxy width")
    if any(not math.isfinite(v) or v <= 0 for v in args.learning_rates):
        raise ValueError("learning rates must be finite and positive")
    if any(not math.isfinite(v) or v < 0 for v in args.weight_decays):
        raise ValueError("weight decays must be finite and nonnegative")
    train_path, val_path = args.cache / "train.bin", args.cache / "val.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Expected {train_path} and {val_path}")

    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from llm_training.models.config import GPTConfig
    from llm_training.models.gpt_mlx import GPT
    from llm_training.pretraining_mlx.data import get_batch
    from llm_training.pretraining_mlx.train import (
        TrainConfig, accumulate_gradients, choose_device, evaluate, make_optimizer,
        learning_rate_at_step,
    )

    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    train_tokens = np.memmap(train_path, dtype="<u4", mode="r",
                             shape=(manifest["train"]["tokens"],))
    val_tokens = np.memmap(val_path, dtype="<u4", mode="r",
                           shape=(manifest["val"]["tokens"],))
    mx.set_default_device(choose_device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.output.exists():
        existing = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()]
    settings = {"seed": args.seed, "cache": str(args.cache.resolve()),
                "vocab_size": args.vocab_size, "context_length": args.context_length,
                "layers": args.layers, "eval_every": args.eval_every, "device": args.device}
    for decay, rate in product(args.weight_decays, args.learning_rates):
        if any(row.get("proxy_width") == args.proxy_width and
               row.get("base_width") == args.base_width and
               row.get("steps") == args.steps and
               row.get("target_width") == args.target_width and
               row.get("batch_size") == args.batch_size and
               all(row.get(key) == value for key, value in settings.items()) and
               row.get("effective_batch_size") == args.batch_size * args.gradient_accumulation_steps and
               row.get("proxy_peak_lr") == rate and row.get("weight_decay") == decay
               for row in existing):
            print(f"Skipping completed LR {rate:g}, WD {decay:g}", flush=True)
            continue
        seed = args.seed
        random.seed(seed)
        mx.random.seed(seed)
        base_width = args.base_width
        config = GPTConfig(vocab_size=args.vocab_size, context_length=args.context_length,
                           n_layers=args.layers, n_heads=4, d_model=args.proxy_width,
                           parametrization="mup", mup_base_width=base_width)
        model = GPT(config)
        training = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                               gradient_accumulation_steps=args.gradient_accumulation_steps,
                               learning_rate=rate, weight_decay=decay,
                               eval_every=args.eval_every, eval_batches=8,
                               seed=seed, sampling="random", lr_schedule="cosine",
                               warmup_steps=max(1, args.steps // 20),
                               min_learning_rate=rate / 10)
        optimizer = make_optimizer(training, config)
        loss_and_grad = nn.value_and_grad(model, lambda x, y: model(x, y)[1])
        generator = random.Random(seed)
        initial_val = evaluate(model, val_tokens, training)
        best_val = initial_val
        evaluations = [{"step": 0, "validation_loss": initial_val}]
        tokens_per_step = training.effective_batch_size * args.context_length
        training_seconds = 0.0
        last_eval_step = 0
        print(f"LR {rate:.2g}, WD {decay:g}: step 0/{args.steps}, val={initial_val:.4f}", flush=True)
        interval_start = perf_counter()
        for step in range(1, args.steps + 1):
            x, y = get_batch(train_tokens, training.effective_batch_size,
                             args.context_length, generator)
            loss, gradients = accumulate_gradients(loss_and_grad, x, y, args.batch_size)
            gradients, _ = optim.clip_grad_norm(gradients, 1.0)
            optimizer.learning_rate = learning_rate_at_step(training, step)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            if step % args.eval_every == 0 or step == args.steps:
                interval_seconds = perf_counter() - interval_start
                training_seconds += interval_seconds
                tokens_per_second = (step - last_eval_step) * tokens_per_step / interval_seconds
                val = evaluate(model, val_tokens, training)
                best_val = min(best_val, val)
                evaluations.append({"step": step, "validation_loss": val, "train_loss": loss,
                                    "train_tokens_per_second": tokens_per_second})
                print(f"LR {rate:.2g}, WD {decay:g}: step {step}/{args.steps}, "
                      f"train={loss:.4f}, val={val:.4f}, best={best_val:.4f}, "
                      f"{tokens_per_second:,.0f} train tok/s",
                      flush=True)
                last_eval_step = step
                interval_start = perf_counter()
        target_rate = rate  # μTransfer keeps the base LR; the optimizer scales matrix LR.
        row = {"proxy_width": args.proxy_width, "target_width": args.target_width,
               "base_width": base_width, "proxy_peak_lr": rate,
               "transferred_target_base_peak_lr": target_rate,
               "transferred_target_matrix_peak_lr": rate * base_width / args.target_width,
               "matrix_lr_multiplier_at_proxy": base_width / args.proxy_width,
               "matrix_lr_multiplier_at_target": base_width / args.target_width,
               "weight_decay": decay, "steps": args.steps,
               "batch_size": args.batch_size,
               "gradient_accumulation_steps": args.gradient_accumulation_steps,
               "effective_batch_size": training.effective_batch_size,
               "train_tokens_per_second": args.steps * tokens_per_step / training_seconds,
               "initial_validation_loss": initial_val, "best_validation_loss": best_val,
               "final_validation_loss": val, "evaluations": evaluations, **settings}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with args.output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        existing.append(row)


if __name__ == "__main__":
    main()
