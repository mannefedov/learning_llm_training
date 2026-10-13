"""Run small GPT pretraining from a TOML config."""

import argparse
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def train_from_config(config_path: Path) -> Path:
    load_dotenv(Path.cwd() / ".env", override=False)
    import torch
    from tokenizers import Tokenizer

    from llm_training.models.gpt import GPT, GPTConfig
    from llm_training.tokenization.data import iter_dataset_texts
    from .data import encode_documents
    from .train import TrainConfig, train

    config_text = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    if set(config) != {"data", "tokenizer", "model", "training", "output"}:
        raise ValueError("Config must contain data, tokenizer, model, training, and output tables")
    output_dir = Path(config["output"]["directory"])
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    model_config = GPTConfig(vocab_size=tokenizer.get_vocab_size(), **config["model"])
    train_config = TrainConfig(**config["training"])
    eos_token = config["tokenizer"]["eos_token"]
    # Train and validation use separate document splits, before token packing.
    data_config = dict(config["data"])
    train_documents = data_config.pop("train_documents")
    val_documents = data_config.pop("val_documents")
    train_split = data_config.pop("train_split")
    val_split = data_config.pop("val_split")
    if train_split == val_split:
        raise ValueError("Use separate document splits for training and validation")
    print("Preparing training tokens...")
    train_tokens = encode_documents(iter_dataset_texts(
        **data_config, split=train_split, num_documents=train_documents
    ), tokenizer, eos_token)
    print("Preparing validation tokens...")
    val_tokens = encode_documents(iter_dataset_texts(
        **data_config, split=val_split, num_documents=val_documents
    ), tokenizer, eos_token)
    for name, tokens in (("train", train_tokens), ("validation", val_tokens)):
        if tokens.numel() <= model_config.context_length:
            raise ValueError(f"{name} data needs at least context_length + 1 tokens")
    print(f"Tokens: train={train_tokens.numel():,}; validation={val_tokens.numel():,}")

    torch.manual_seed(train_config.seed)
    model = GPT(model_config)
    optimizer = train(model, train_tokens, val_tokens, train_config)

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "model_config": asdict(model_config),
        "optimizer": optimizer.state_dict(),
        "step": train_config.steps,
        "config": config,
    }, checkpoint_path)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    (output_dir / "config.toml").write_text(config_text, encoding="utf-8")
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small GPT model")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
