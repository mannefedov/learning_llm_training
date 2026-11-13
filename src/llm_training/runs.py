"""Allocate fresh output directories without overwriting previous runs."""

from pathlib import Path
import re


def create_run_directory(requested: Path) -> Path:
    """Create requested, or increment its trailing number until a path is free.

    For example, run-01 becomes run-02; an unnumbered name becomes name-02.
    mkdir is exclusive so concurrent invocations cannot claim the same run.
    """
    match = re.search(r"(\d+)$", requested.name)
    prefix = requested.name[:match.start()] if match else requested.name + "-"
    width = len(match.group()) if match else 2
    number = int(match.group()) if match else 1
    candidate = requested
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            number += 1
            candidate = requested.with_name(f"{prefix}{number:0{width}d}")
