"""Run small GPT pretraining from a TOML config."""

import argparse
from dataclasses import asdict
from pathlib import Path

from llm_training.runs import create_run_directory

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def train_from_config(config_path: Path) -> Path:
    load_dotenv(Path.cwd() / ".env", override=False)
    import json
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer

    from llm_training.models.gpt_mlx import GPT, GPTConfig
    from llm_training.tokenization.data import iter_dataset_texts
    from .data import encode_documents
    from .train import TrainConfig, choose_device, train
    from .tracking import environment_metadata, log_run
    from .sampling import GenerationConfig, log_samples

    config_text = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    required = {"data", "tokenizer", "model", "training", "output"}
    if not required <= set(config) or set(config) - required - {"generation", "cache"}:
        raise ValueError("Config requires data, tokenizer, model, training, output; generation and cache are optional")
    output_dir = Path(config["output"]["directory"])
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    model_config = GPTConfig(vocab_size=tokenizer.get_vocab_size(), **config["model"])
    train_config = TrainConfig(**config["training"])
    generation_config = GenerationConfig(**config.get("generation", {}))
    eos_token = config["tokenizer"]["eos_token"]
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f"Tokenizer has no EOS token {eos_token!r}")
    if generation_config.enabled:
        for prompt in generation_config.prompts:
            if not tokenizer.encode(prompt, add_special_tokens=False).ids:
                raise ValueError("Generation prompts must encode to at least one token")
    # Train and validation use separate document splits, before token packing.
    data_config = dict(config["data"])
    train_documents = data_config.pop("train_documents")
    val_documents = data_config.pop("val_documents")
    train_split = data_config.pop("train_split")
    val_split = data_config.pop("val_split")
    if train_split == val_split:
        raise ValueError("Use separate document splits for training and validation")
    cached_tokens = None
    if "cache" in config:
        from llm_training.pretraining.token_cache import cache_identity, open_cache
        identity = cache_identity(config["data"], Path(config["tokenizer"]["path"]), eos_token)
        cache_dir = Path(config["cache"]["directory"])
        if not (cache_dir / "manifest.json").exists():
            raise FileNotFoundError("Prepare the token cache first: python -m "
                                    f"llm_training.pretraining_mlx.prepare --config {config_path}")
        train_array, val_array, cache_manifest = open_cache(cache_dir, identity)
        cached_tokens = train_array, val_array
    if train_config.sampling == "sequential" and cached_tokens is None:
        raise ValueError("Sequential training requires a prepared [cache] directory")
    if cached_tokens is not None:
        train_tokens, val_tokens = cached_tokens
        token_budget = train_config.batch_size * model_config.context_length
        available_steps = (train_tokens.size - 1) // token_budget
        if train_config.sampling == "sequential":
            if train_config.steps > available_steps:
                raise ValueError(f"Only {available_steps:,} full sequential steps available")
            covered = train_config.steps * token_budget
            print(f"Sequential coverage: {covered:,}/{train_tokens.size - 1:,} target tokens "
                  f"({covered / (train_tokens.size - 1):.2%}); full pass: {available_steps:,} steps")
    mx.set_default_device(choose_device(train_config.device))
    mx.random.seed(train_config.seed)
    model = GPT(model_config)
    output_dir = create_run_directory(output_dir)
    print(f"Run directory: {output_dir}")
    (output_dir / "config.toml").write_text(config_text, encoding="utf-8")
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    metadata = {
        **environment_metadata(tokenizer_path),
        "backend": "mlx",
        "run_name": output_dir.name,
        "model_config": asdict(model_config),
        "training_config": asdict(train_config),
        "generation_config": asdict(generation_config),
        "data_config": config["data"],
        "device": "gpu" if choose_device(train_config.device) == mx.gpu else "cpu",
        "parameter_count": sum(p.size for _, p in tree_flatten(model.parameters())),
    }
    if cached_tokens is not None:
        metadata["token_cache"] = {"directory": str(cache_dir.resolve()), **cache_manifest}
    checkpoint_path = output_dir / "checkpoint.json"
    with log_run(output_dir, metadata) as record_metrics:
        if cached_tokens is not None:
            train_tokens, val_tokens = cached_tokens
        else:
            print("Preparing training tokens...")
            train_tokens = encode_documents(iter_dataset_texts(
                **data_config, split=train_split, num_documents=train_documents
            ), tokenizer, eos_token)
            print("Preparing validation tokens...")
            val_tokens = encode_documents(iter_dataset_texts(
                **data_config, split=val_split, num_documents=val_documents
            ), tokenizer, eos_token)
        for name, tokens in (("train", train_tokens), ("validation", val_tokens)):
            if tokens.size <= model_config.context_length:
                raise ValueError(f"{name} data needs at least context_length + 1 tokens")
        print(f"Tokens: train={train_tokens.size:,}; validation={val_tokens.size:,}")
        with log_samples(output_dir, model, tokenizer, eos_id,
                         generation_config, train_config.steps) as record_samples:
            optimizer = train(model, train_tokens, val_tokens, train_config,
                              record_metrics, record_samples)
        model.save_weights(str(output_dir / "model.safetensors"))
        mx.save_safetensors(str(output_dir / "optimizer.safetensors"),
                            dict(tree_flatten(optimizer.state)))
        checkpoint_path.write_text(json.dumps({
            "backend": "mlx",
            "model": "model.safetensors",
            "optimizer": "optimizer.safetensors",
            "model_config": asdict(model_config),
            "step": train_config.steps,
            "config": config,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small GPT model with MLX")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
