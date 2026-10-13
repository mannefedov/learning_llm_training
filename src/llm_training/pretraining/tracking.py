"""Small, local CSV and JSON records for pretraining runs."""

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess


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
            package: version(package) for package in ("torch", "tokenizers", "datasets")
        },
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }


@contextmanager
def log_run(output_dir: Path, metadata: dict):
    """Yield a loss-recording callback; flush each row and record run status.

    Losses have the same meaning and cadence as the existing console output.
    At step zero there is no training loss, represented by an empty CSV field.
    """
    metadata = dict(metadata)
    metadata.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
    metadata_path = output_dir / "metadata.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    save_metadata()
    try:
        with (output_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["step", "train_loss", "val_loss"])
            writer.writeheader()
            file.flush()

            def record(step, train_loss, val_loss):
                writer.writerow({"step": step, "train_loss": train_loss, "val_loss": val_loss})
                file.flush()

            yield record
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        raise
    else:
        metadata["status"] = "completed"
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_metadata()
