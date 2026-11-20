"""Run small GPT pretraining from a TOML config."""

import argparse
from dataclasses import asdict, replace
import math
from pathlib import Path

from llm_training.runs import create_run_directory

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def train_from_config(config_path: Path, resume: Path | None = None,
                      *, anneal_steps: int | None = None, new_cache: bool = False,
                      allow_accumulation_change: bool = False,
                      anneal_schedule: str | None = None) -> Path:
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
    from .statistics import StatisticsConfig, log_statistics
    from .checkpoint import (CheckpointConfig, CheckpointWriter, checkpoint_tokens,
                             read_checkpoint, restore_checkpoint)

    config_text = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    required = {"data", "tokenizer", "model", "training", "output"}
    optional = {"generation", "cache", "checkpoint", "statistics", "resume_anneal"}
    if not required <= set(config) or set(config) - required - optional:
        raise ValueError("Config requires data, tokenizer, model, training and output; "
                         "optional sections: " + ", ".join(sorted(optional)))
    output_dir = Path(config["output"]["directory"])
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    model_config = GPTConfig(vocab_size=tokenizer.get_vocab_size(), **config["model"])
    train_config = TrainConfig(**config["training"])
    anneal = config.get("resume_anneal")
    if anneal_schedule is not None and (anneal_steps is None or anneal_schedule not in ("cosine", "staircase")):
        raise ValueError("--anneal-schedule requires --anneal-steps and cosine or staircase")
    if allow_accumulation_change and (not new_cache or anneal_steps is None):
        raise ValueError("--allow-accumulation-change requires --resume-new-cache and --anneal-steps")
    if new_cache and (resume is None or anneal is not None):
        raise ValueError("--resume-new-cache requires --resume with a new continuation config")
    if (anneal_steps is not None or anneal is not None) and resume is None:
        raise ValueError("Annealing requires --resume")
    if anneal_steps is not None and (type(anneal_steps) is not int or anneal_steps < 2):
        raise ValueError("--anneal-steps must be at least 2")
    if anneal_steps is not None and anneal is not None:
        raise ValueError("Use the original training config to start a new anneal branch")
    data_start_step = 0
    if resume is not None:
        resume, saved = read_checkpoint(resume)
        data_start_step = saved["step"] if new_cache else saved.get("data_start_step", 0)
    if anneal_steps is not None or anneal is not None:
        origin = saved["step"] if anneal is None else anneal["checkpoint_step"]
        length = anneal_steps if anneal is None else anneal["steps"]
        if not origin <= saved["step"] < origin + length:
            raise ValueError("Resume checkpoint step is outside the anneal window")
        schedule = (anneal_schedule or
                    ("staircase" if train_config.lr_schedule == "staircase" else "cosine"))
        if anneal is not None:
            schedule = anneal.get("schedule", "cosine")
        train_config = replace(
            train_config, steps=origin + length,
            lr_schedule=schedule,
            lr_decay=True,
            warmup_steps=0,
            lr_decay_start_step=origin + 1, lr_decay_steps=origin + length,
            min_learning_rate=0.0,
            learning_rate=train_config.learning_rate if anneal is None else anneal["start_lr"],
        )
        if anneal_steps is not None:
            output_dir = output_dir.with_name(f"{output_dir.name}-anneal-{length}-01")
        elif saved["config"].get("resume_anneal") != anneal:
            raise ValueError("Resume checkpoint belongs to a different anneal schedule")
    checkpoint_config = CheckpointConfig(**config.get("checkpoint", {}))
    generation_config = GenerationConfig(**config.get("generation", {}))
    statistics_config = StatisticsConfig(**config.get("statistics", {}))
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
            epoch_steps = batches_per_epoch(train_tokens, train_config.effective_batch_size, model_config.context_length)
            print(f"Sequential training: {epoch_steps:,} steps per epoch; "
                  f"target {(train_config.steps - data_start_step) / epoch_steps:.2f} epochs "
                  f"on this cache (absolute step {train_config.steps:,})")
    mx.set_default_device(choose_device(train_config.device))
    mx.random.seed(train_config.seed)
    model = GPT(model_config)
    optimizer = make_optimizer(train_config, model_config)
    start_step, batch_rng_state = 0, None
    tokens_before_start = 0
    cache_signature = cache_manifest if cached_tokens is not None else None
    if resume is not None:
        tokens_before_start = checkpoint_tokens(saved)
        start_step, batch_rng_state = restore_checkpoint(
            resume, model, optimizer, config, train_config,
            Path(config["tokenizer"]["path"]), cache_signature,
            keep_saved_lr=anneal_steps is not None, new_cache=new_cache,
            allow_accumulation_change=allow_accumulation_change)
        print(f"Resuming from step {start_step:,}; target: {train_config.steps:,}")
    if anneal_steps is not None:
        start_lr = optimizer.learning_rate.item()
        if not math.isfinite(start_lr) or start_lr <= 0:
            raise ValueError("Annealing requires a positive finite LR in the source checkpoint")
        train_config = replace(train_config, learning_rate=start_lr)
        anneal = {"checkpoint_step": start_step, "steps": anneal_steps, "start_lr": start_lr}
        if train_config.lr_schedule == "staircase":
            anneal["schedule"] = "staircase"
        config["resume_anneal"] = anneal
        config_text += (f"\n[resume_anneal]\ncheckpoint_step = {start_step}\n"
                        f"steps = {anneal_steps}\nstart_lr = {start_lr!r}\n")
        if "schedule" in anneal:
            config_text += 'schedule = "staircase"\n'
    if anneal is not None:
        print(f"{train_config.lr_schedule.capitalize()} anneal: "
              f"{anneal['start_lr']:.8g} → 0 over {anneal['steps']:,} updates")
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
        "effective_batch_size": train_config.effective_batch_size,
        "tokens_per_update": train_config.effective_batch_size * model_config.context_length,
        "generation_config": asdict(generation_config),
        "checkpoint_config": asdict(checkpoint_config),
        "statistics_config": asdict(statistics_config),
        "start_step": start_step,
        "data_start_step": data_start_step,
        "resumed_from": str(resume.resolve()) if resume is not None else None,
        "data_config": config["data"],
        "device": "gpu" if choose_device(train_config.device) == mx.gpu else "cpu",
        "parameter_count": sum(p.size for _, p in tree_flatten(model.parameters())),
    }
    if anneal is not None:
        metadata["resume_anneal"] = anneal
    if train_config.sampling == "sequential":
        metadata["steps_per_epoch"] = epoch_steps
        metadata["target_epochs"] = (train_config.steps - data_start_step) / epoch_steps
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
                         generation_config, train_config.steps) as record_samples, log_statistics(
                             output_dir, model, train_tokens, val_tokens,
                             train_config, statistics_config) as statistics:
            writer = CheckpointWriter(output_dir, model, config, train_config,
                                      checkpoint_config, cache_signature, data_start_step=data_start_step,
                                      start_step=start_step, tokens_before_start=tokens_before_start)
            train(model, train_tokens, val_tokens, train_config, record_metrics, record_samples,
                  optimizer=optimizer, start_step=start_step, batch_rng_state=batch_rng_state,
                  record_checkpoint=writer, statistics=statistics, data_start_step=data_start_step,
                  tokens_before_start=tokens_before_start)
        print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small GPT model with MLX")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path,
                        help="Run directory or checkpoint.json to restore; steps is the total target")
    parser.add_argument("--anneal-steps", type=int,
                        help="With --resume, run this many additional updates and cosine decay the saved LR to zero")
    parser.add_argument("--resume-new-cache", action="store_true",
                        help="With --resume, use the new sequential cache from its first batch")
    parser.add_argument("--allow-accumulation-change", action="store_true",
                        help="Allow a different accumulation count in a new-cache anneal branch")
    parser.add_argument("--anneal-schedule", choices=("cosine", "staircase"),
                        help="Schedule for a new --anneal-steps branch")
    args = parser.parse_args()
    train_from_config(args.config, resume=args.resume, anneal_steps=args.anneal_steps,
                      new_cache=args.resume_new_cache,
                      allow_accumulation_change=args.allow_accumulation_change,
                      anneal_schedule=args.anneal_schedule)


if __name__ == "__main__":
    main()
