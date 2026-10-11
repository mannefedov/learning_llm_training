"""Run tokenizer training with a TOML config.

Usage: python -m llm_training.tokenization --config configs/tokenization/fineweb2_bpe.toml
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def train_from_config(config_path: Path) -> Path:
    """Train and save a tokenizer and its config in a new output directory.

    Relative output paths are resolved against the current working directory.
    An existing directory is rejected to avoid overwriting a previous run.
    """
    # Load credentials before importing the Hugging Face libraries.
    # Already exported environment variables take precedence over .env.
    load_dotenv(Path.cwd() / ".env", override=False)
    from .bpe import train_byte_level_bpe
    from .data import iter_dataset_texts

    config_text = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    if set(config) != {"data", "tokenizer", "output"}:
        raise ValueError("Config must contain only data, tokenizer, and output tables")
    if set(config["output"]) != {"directory"}:
        raise ValueError("The output table must contain only directory")

    output_dir = Path(config["output"]["directory"])
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")

    texts = iter_dataset_texts(**config["data"])
    tokenizer = train_byte_level_bpe(texts, **config["tokenizer"])

    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    (output_dir / "config.toml").write_text(config_text, encoding="utf-8")
    print(f"Saved {tokenizer.get_vocab_size()} tokens to {tokenizer_path}")
    return tokenizer_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument("--config", type=Path, required=True, help="TOML config path")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
