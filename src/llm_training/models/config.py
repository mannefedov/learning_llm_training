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
    stable_softmax: bool = False
    attention_backend: str = "reference"
    parametrization: str = "standard"
    mup_base_width: int | None = None

    def __post_init__(self):
        if self.attention_backend not in ("reference", "mlx_fused", "mlx_fused_window"):
            raise ValueError("attention_backend must be reference, mlx_fused, or mlx_fused_window")
        if self.attention_backend in ("mlx_fused", "mlx_fused_window"):
            if self.stable_softmax:
                raise ValueError("mlx_fused uses native softmax; set stable_softmax=false")
            if (self.attention_backend == "mlx_fused" and self.attention_types
                    and "sliding_window" in self.attention_types):
                raise ValueError("mlx_fused supports full attention only")
        if self.parametrization not in ("standard", "mup"):
            raise ValueError("parametrization must be standard or mup")
        if self.parametrization == "mup":
            if (not isinstance(self.mup_base_width, int) or isinstance(self.mup_base_width, bool)
                    or self.mup_base_width < 1):
                raise ValueError("mup_base_width must be a positive integer with parametrization='mup'")
        if not isinstance(self.stable_softmax, bool):
            raise ValueError("stable_softmax must be a boolean")
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
