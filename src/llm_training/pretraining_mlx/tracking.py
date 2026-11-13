"""Small, local CSV and JSON records for pretraining runs."""

import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess

from llm_training.pretraining.tracking import log_run


def environment_metadata(tokenizer_path: Path) -> dict:
    """Record code/version information and tokenizer identity, without secrets."""
    def git_output(*args):
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    status = git_output("status", "--porcelain")
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": {
            package: version(package) for package in ("mlx", "tokenizers", "datasets")
        },
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }

