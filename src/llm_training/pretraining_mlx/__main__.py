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


def train_from_config(config_path: Path, resume: Path | None = None) -> Path:
    load_dotenv(Path.cwd() / ".env", override=False)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer

    from llm_training.models.gpt_mlx import GPT, GPTConfig
    from llm_training.tokenization.data import iter_dataset_texts
    from .data import encode_documents, batches_per_epoch
    from .train import TrainConfig, choose_device, train, make_optimizer
    from .tracking import environment_metadata, log_run
    from .sampling import GenerationConfig, log_samples
    from .checkpoint import CheckpointConfig, CheckpointWriter, restore_checkpoint

    config_text = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    required = {"data", "tokenizer", "model", "training", "output"}
    if not required <= set(config) or set(config) - required - {"generation", "cache", "checkpoint"}:
        raise ValueError("Config requires data, tokenizer, model, training, output; generation, cache, checkpoint are optional")
    output_dir = Path(config["output"]["directory"])
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    model_config = GPTConfig(vocab_size=tokenizer.get_vocab_size(), **config["model"])
    train_config = TrainConfig(**config["training"])
    checkpoint_config = CheckpointConfig(**config.get("checkpoint", {}))
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
        if train_config.sampling == "sequential":
            epoch_steps = batches_per_epoch(train_tokens, train_config.batch_size, model_config.context_length)
            print(f"Sequential training: {epoch_steps:,} steps per epoch; "
                  f"target {train_config.steps / epoch_steps:.2f} epochs ({train_config.steps:,} steps)")
    mx.set_default_device(choose_device(train_config.device))
    mx.random.seed(train_config.seed)
    model = GPT(model_config)
    optimizer = make_optimizer(train_config)
    start_step, batch_rng_state = 0, None
    cache_signature = cache_manifest if cached_tokens is not None else None
    if resume is not None:
        start_step, batch_rng_state = restore_checkpoint(
            resume, model, optimizer, config, train_config,
            Path(config["tokenizer"]["path"]), cache_signature)
        print(f"Resuming from step {start_step:,}; target: {train_config.steps:,}")
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
        "checkpoint_config": asdict(checkpoint_config),
        "start_step": start_step,
        "resumed_from": str(resume.resolve()) if resume is not None else None,
        "data_config": config["data"],
        "device": "gpu" if choose_device(train_config.device) == mx.gpu else "cpu",
        "parameter_count": sum(p.size for _, p in tree_flatten(model.parameters())),
    }
    if train_config.sampling == "sequential":
        metadata["steps_per_epoch"] = epoch_steps
        metadata["target_epochs"] = train_config.steps / epoch_steps
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
            writer = CheckpointWriter(output_dir, model, config, train_config,
                                      checkpoint_config, cache_signature)
            train(model, train_tokens, val_tokens, train_config, record_metrics, record_samples,
                  optimizer=optimizer, start_step=start_step, batch_rng_state=batch_rng_state,
                  record_checkpoint=writer)
        print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small GPT model with MLX")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path,
                        help="Run directory or checkpoint.json to restore; steps is the total target")
    args = parser.parse_args()
    train_from_config(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
