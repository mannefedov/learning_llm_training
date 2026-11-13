"""Periodic qualitative samples saved alongside an MLX training run."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from pathlib import Path

from .generate import generate_text


@dataclass
class GenerationConfig:
    enabled: bool = True
    every: int = 500
    prompts: list[str] = field(default_factory=lambda: ["Сегодня утром я"])
    max_new_tokens: int = 100
    temperature: float = 0.8
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("generation.enabled must be a boolean")
        if self.every < 1 or self.max_new_tokens < 1:
            raise ValueError("generation.every and max_new_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("generation.temperature must be finite and positive")
        if (not isinstance(self.prompts, list) or not self.prompts
                or any(not isinstance(p, str) or not p for p in self.prompts)):
            raise ValueError("generation.prompts must be a nonempty list of nonempty strings")


@contextmanager
def log_samples(output_dir: Path, model, tokenizer, eos_id: int,
                config: GenerationConfig, total_steps: int):
    """Yield a callback writing samples at step zero, every N steps, and the end.

    Each completed sample is flushed, preserving it after an interrupted run.
    Sampling uses local PRNG keys and restores the model's previous train mode.
    """
    if not config.enabled:
        yield None
        return
    with (output_dir / "samples.txt").open("x", encoding="utf-8") as file:
        def record(step):
            if step % config.every != 0 and step != total_steps:
                return
            was_training = model.training
            model.eval()
            try:
                for prompt in config.prompts:
                    text = generate_text(model, tokenizer, prompt,
                                         max_new_tokens=config.max_new_tokens,
                                         temperature=config.temperature,
                                         seed=config.seed, eos_id=eos_id)
                    file.write(f"=== step={step} temperature={config.temperature} "
                               f"seed={config.seed} max_new_tokens={config.max_new_tokens} ===\n"
                               f"Prompt: {prompt!r}\n{text}\n\n")
                    file.flush()
            finally:
                model.train(was_training)
        yield record
