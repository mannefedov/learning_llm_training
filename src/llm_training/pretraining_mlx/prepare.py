"""Prepare the disk token cache selected by an MLX training config."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def main():
    parser = argparse.ArgumentParser(description="Stream and tokenize a corpus to disk")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env", override=False)
    from llm_training.pretraining.token_cache import prepare_cache

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    token_limit = (config["training"]["steps"] * config["training"]["batch_size"]
                   * config["model"]["context_length"] + 1)
    prepare_cache(Path(config["cache"]["directory"]), config["data"],
                  Path(config["tokenizer"]["path"]), config["tokenizer"]["eos_token"], token_limit,
                  allow_short_train=config["training"].get("sampling", "random") == "sequential")
    print(f"Cache ready: {config['cache']['directory']}", flush=True)


if __name__ == "__main__":
    main()
    # Only the standalone CLI takes this path, after successful preparation
    # and closed output files. Arrow 26 can deadlock in its native destructors
    # even after Python cleanup: https://github.com/apache/arrow/issues/45214
    # Avoid those destructors; never mask a preparation exception with success.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
