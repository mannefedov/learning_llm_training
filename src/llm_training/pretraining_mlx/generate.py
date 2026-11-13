"""Sample text from a saved MLX run without loading training data or optimizer."""

import argparse
import json
import math
from pathlib import Path


def load_checkpoint(run_dir: Path, device: str = "auto"):
    """Load the model and the tokenizer saved together in a run directory."""
    import mlx.core as mx
    from tokenizers import Tokenizer
    from llm_training.models.gpt_mlx import GPT, GPTConfig
    from .train import choose_device

    mx.set_default_device(choose_device(device))
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint["backend"] != "mlx":
        raise ValueError("Expected an MLX checkpoint")
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_weights(str(run_dir / checkpoint["model"]))
    model.eval()
    mx.eval(model.parameters())
    tokenizer = Tokenizer.from_file(str(run_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match the model")
    eos_token = checkpoint["config"]["tokenizer"]["eos_token"]
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f"Tokenizer has no EOS token {eos_token!r}")
    return model, tokenizer, eos_id


def generate_text(model, tokenizer, prompt: str, *, max_new_tokens: int = 100,
                  temperature: float = 0.8, seed: int = 42,
                  eos_id: int | None = None) -> str:
    """Return prompt plus continuation; stop at EOS or the new-token limit.

    Use a local sampling key so comparisons are repeatable on the same backend
    without changing the training RNG. The model crops context using its existing generate().
    """
    import mlx.core as mx

    if not math.isfinite(temperature) or temperature <= 0 or max_new_tokens < 0:
        raise ValueError("temperature must be finite and positive; max_new_tokens nonnegative")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids:
        raise ValueError("Provide a nonempty prompt that encodes to at least one token")
    key = mx.random.key(seed)
    tokens = mx.array([ids], dtype=mx.int32)
    for _ in range(max_new_tokens):
        key, sample_key = mx.random.split(key)
        tokens = model.generate(tokens, max_new_tokens=1, temperature=temperature, key=sample_key)
        if eos_id is not None and tokens[0, -1].item() == eos_id:
            break
    return tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Sample a saved MLX GPT checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True,
                        help="Repeat this option to compare multiple prompts")
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.8])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()
    if args.max_new_tokens < 0 or any(not math.isfinite(t) or t <= 0 for t in args.temperature):
        parser.error("max-new-tokens must be nonnegative and temperatures finite and positive")
    if any(not prompt for prompt in args.prompt):
        parser.error("prompts must be nonempty")
    model, tokenizer, eos_id = load_checkpoint(args.run_dir, args.device)
    print(f"Checkpoint: {args.run_dir}")
    for prompt in args.prompt:
        for temperature in args.temperature:
            print(f"\nPrompt: {prompt!r}; temperature={temperature}; seed={args.seed}")
            print(generate_text(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
                                temperature=temperature, seed=args.seed, eos_id=eos_id))


if __name__ == "__main__":
    main()
