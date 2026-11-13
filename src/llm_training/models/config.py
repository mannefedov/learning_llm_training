"""Backend-independent GPT dimensions and per-layer attention choices."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    attention_types: list[str] | None = None
    sliding_window: int | None = None

    def __post_init__(self):
        if self.attention_types is not None:
            if (not isinstance(self.attention_types, list)
                    or len(self.attention_types) != self.n_layers
                    or any(t not in ("full", "sliding_window") for t in self.attention_types)):
                raise ValueError("attention_types must list full/sliding_window once per layer")
        if self.sliding_window is not None and (
            not isinstance(self.sliding_window, int) or isinstance(self.sliding_window, bool)
            or self.sliding_window < 1
        ):
            raise ValueError("sliding_window must be a positive integer")
        if self.attention_types and "sliding_window" in self.attention_types and self.sliding_window is None:
            raise ValueError("Sliding-window layers require sliding_window")
        if min(self.vocab_size, self.context_length, self.n_layers,
               self.n_heads, self.d_model) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")

    def attention_type(self, layer_index: int) -> str:
        return "full" if self.attention_types is None else self.attention_types[layer_index]
