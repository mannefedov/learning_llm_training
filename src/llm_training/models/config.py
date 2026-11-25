"""Backend-independent GPT dimensions and per-layer attention choices."""

from dataclasses import dataclass
import math


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
    position_encoding: str = "learned"
    attention_sinks: bool = False
    num_experts: int = 0
    experts_per_token: int = 2
    expert_hidden_dim: int | None = None
    moe_aux_loss_coef: float = 0.01
    attention_architecture: str = "mha"
    mla_q_rank: int = 32
    mla_kv_rank: int = 16
    mla_qk_head_dim: int = 16
    mla_rope_head_dim: int = 8
    mla_v_head_dim: int = 16

    def __post_init__(self):
        if self.attention_architecture not in ("mha", "mla"):
            raise ValueError("attention_architecture must be mha or mla")
        for name in ("mla_q_rank", "mla_kv_rank", "mla_qk_head_dim",
                     "mla_rope_head_dim", "mla_v_head_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mla_rope_head_dim % 2:
            raise ValueError("mla_rope_head_dim must be even")
        if self.attention_architecture == "mla":
            if self.position_encoding != "rope":
                raise ValueError("MLA requires position_encoding='rope'")
            if self.parametrization != "standard":
                raise ValueError("The first MLA implementation supports standard parameterization")
            if self.attention_sinks:
                raise ValueError("MLA currently does not support attention sinks")
            if self.attention_backend in ("mlx_fused", "mlx_fused_window"):
                qk_dim = self.mla_qk_head_dim + self.mla_rope_head_dim
                if qk_dim not in (64, 128) or self.mla_v_head_dim != qk_dim:
                    raise ValueError("Fused MLA requires equal QK/value head dimensions of 64 or 128")
        if type(self.num_experts) is not int or self.num_experts < 0:
            raise ValueError("num_experts must be a nonnegative integer (0 selects dense FFNs)")
        if type(self.experts_per_token) is not int or self.experts_per_token < 1:
            raise ValueError("experts_per_token must be a positive integer")
        if self.num_experts and self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token must not exceed num_experts")
        if self.num_experts and self.parametrization != "standard":
            raise ValueError("The first MoE implementation supports standard parameterization")
        if self.expert_hidden_dim is not None and (
                type(self.expert_hidden_dim) is not int or self.expert_hidden_dim < 1):
            raise ValueError("expert_hidden_dim must be a positive integer")
        if not math.isfinite(self.moe_aux_loss_coef) or self.moe_aux_loss_coef < 0:
            raise ValueError("moe_aux_loss_coef must be finite and nonnegative")
        if not isinstance(self.attention_sinks, bool):
            raise ValueError("attention_sinks must be a boolean")
        if self.attention_sinks and self.attention_backend != "reference":
            raise ValueError("attention_sinks currently requires reference attention")
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
        if self.position_encoding not in ("learned", "rope"):
            raise ValueError("position_encoding must be learned or rope")
        if (self.position_encoding == "rope" and self.attention_architecture == "mha"
                and (self.d_model // self.n_heads) % 2):
            raise ValueError("RoPE requires an even attention head dimension")

    def attention_type(self, layer_index: int) -> str:
        return "full" if self.attention_types is None else self.attention_types[layer_index]
