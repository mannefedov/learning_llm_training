"""Periodic, bounded checkpoint storage and explicit training-state restoration."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from llm_training.models.config import GPTConfig


@dataclass
class CheckpointConfig:
    every: int = 1000
    keep: int = 1

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (self.every, self.keep)):
            raise ValueError("Checkpoint every/keep must be integers")
        if self.every < 0 or self.keep < 1:
            raise ValueError("checkpoint.every must be nonnegative and keep must be positive")


def read_checkpoint(path: Path):
    path = path / "checkpoint.json" if path.is_dir() else path
    return path, json.loads(path.read_text(encoding="utf-8"))


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def restore_checkpoint(path, model, optimizer, config, train_config, tokenizer_path, cache):
    """Restore only compatible runs; steps is the new total, not additional steps."""
    path, saved = read_checkpoint(Path(path))
    if saved.get("format_version") != 2:
        raise ValueError("This checkpoint predates resume support; it can still be used for generation")
    if asdict(GPTConfig(**saved["model_config"])) != asdict(model.config):
        raise ValueError("Resume model configuration differs from the checkpoint")
    old_training = saved["training_config"]
    for key in ("batch_size", "learning_rate", "weight_decay", "grad_clip", "seed", "sampling"):
        if old_training[key] != getattr(train_config, key):
            raise ValueError(f"Resume training setting differs: {key}")
    if saved["config"]["data"] != config["data"] or saved["token_cache"] != cache:
        raise ValueError("Resume data/cache differs from the checkpoint")
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if saved["tokenizer_sha256"] != digest or saved["config"]["tokenizer"]["eos_token"] != config["tokenizer"]["eos_token"]:
        raise ValueError("Resume tokenizer differs from the checkpoint")
    step = saved["step"]
    if not 0 <= step < train_config.steps:
        raise ValueError("Set training.steps to a total greater than the checkpoint step")
    model.load_weights(str(path.parent / saved["model"]))
    optimizer.state = tree_unflatten(mx.load(str(path.parent / saved["optimizer"])))
    if optimizer.state["step"].item() != step:
        raise ValueError("Checkpoint optimizer step does not match its manifest")
    mx.eval(model.parameters(), optimizer.state)
    return step, _tuples(saved["batch_rng_state"])


class CheckpointWriter:
    def __init__(self, directory, model, config, train_config, settings, cache=None):
        self.directory = directory
        self.model = model
        self.config = config
        self.train_config = train_config
        self.settings = settings
        self.cache = cache
        self.tokenizer_hash = hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest()

    def __call__(self, step, optimizer, generator):
        if step != self.train_config.steps and (not self.settings.every or step % self.settings.every):
            return
        root = self.directory / "checkpoints"
        root.mkdir(exist_ok=True)
        destination = root / f"step-{step:09d}"
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
        try:
            self.model.save_weights(str(temporary / "model.safetensors"))
            mx.save_safetensors(str(temporary / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
            saved = {
                "format_version": 2, "backend": "mlx", "step": step,
                "model": "model.safetensors", "optimizer": "optimizer.safetensors",
                "tokenizer": "../../tokenizer.json", "tokenizer_sha256": self.tokenizer_hash,
                "model_config": asdict(self.model.config), "training_config": asdict(self.train_config),
                "config": self.config, "token_cache": self.cache,
                # Current GPT training has no dropout; its only stochastic input
                # is the host batch RNG. Generation uses independent local keys.
                "batch_rng_state": generator.getstate(),
            }
            (temporary / "checkpoint.json").write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
            temporary.rename(destination)
            # Publish the new pointer only when all checkpoint files are complete.
            latest = {**saved, "tokenizer": "tokenizer.json"}
            for key in ("model", "optimizer"):
                latest[key] = str(destination.relative_to(self.directory) / saved[key])
            pointer = self.directory / ".checkpoint.json.tmp"
            pointer.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
            pointer.replace(self.directory / "checkpoint.json")
            checkpoints = sorted(
                (p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"step-\d+", p.name)),
                key=lambda p: int(p.name[5:]))
            for old in checkpoints[:-self.settings.keep]:
                shutil.rmtree(old)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
