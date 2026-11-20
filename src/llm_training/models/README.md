# GPT models

| File | Backend | Run it |
| --- | --- | --- |
| `gpt.py` | PyTorch | [Training and sampling](../pretraining/README.md) |
| `gpt_mlx.py` | MLX | [Training and sampling](../pretraining_mlx/README.md) |

Both implement a small GPT with causal attention, learned positions, pre-norm
blocks, GELU, and tied embeddings. Dimensions come from `[model]` in the training config.

Architecture reference: [build-nanogpt](https://github.com/karpathy/build-nanogpt).

## Attention variants

Set one type per layer in `[model]` (omitting the list keeps full attention):

```toml
attention_types = ["sliding_window", "full", "sliding_window", "full"]
sliding_window = 64
```

The window includes the current token.

`attention.py` and `attention_mlx.py` share projections within each backend;
`build_attention` selects the variant. SWA tiles the score computation; no KV
cache is used by the reference implementation.

## Corrected softmax experiment

Set `stable_softmax = true` in `[model]` to use corrected attention softmax and
hard-target cross-entropy in either backend. The default stays `false` for
existing runs. A different softmax setting requires a fresh run. Implementation
and source: [numerics.py](numerics.py), [MLX version](numerics_mlx.py),
[corrected reference and erratum](https://jonbarron.info/softmaxfix/).

## Fused full attention (MLX)

`attention_backend = "reference"` and otherwise
identical settings. Existing checkpoints with a different layout cannot resume.

`attention_backend = "mlx_fused"` requires Metal and forces MLX's fused kernel;
unsupported shapes fail instead of falling back. Training sequences must exceed
eight tokens in MLX 0.32.4; shorter prefixes work for generation. It cannot be combined with
`stable_softmax = true` or SWA. The default backend remains `reference`.

[Experiment findings](../../../docs/models/experiment_findings.md) · [Exploration backlog](../../../docs/explorations/README.md)

## Rotary position encoding

Set `position_encoding = "rope"` in `[model]` to rotate query and key
channel pairs by token position. This replaces the learned input position
table, so start from fresh weights. RoPE works with full or sliding-window
attention and requires an even head dimension. Existing configs default to
`"learned"`. [RoFormer](https://arxiv.org/abs/2104.09864)

Run the width-256 μP 150M-token experiment on the existing 2B cache:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_mup_width256_rope_150m.toml
```

It trains for 7,325 updates (150.016M targets) and saves results under
`artifacts/pretraining/gpt-mlx-mup-width256-rope-150m-01`. For a matched
control, copy the run config, set `position_encoding = "learned"`, and give
the control its own output directory. Compare validation at equal steps.

### Smaller attention windows

Compare the full-attention RoPE run with three local layers and one final
full layer at context 256. The window includes the current token.

```sh
for window in 32 16; do
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
    .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
    --config "configs/pretraining/sequential_gpt_mlx_mup_width256_rope_swa${window}_150m.toml" || break
done
```

Each run starts fresh and uses the same 150.016M-target budget and LR schedule
as the all-full RoPE config. Outputs use `rope-swa32-150m` and
`rope-swa16-150m` in their run names. Compare validation and measured tok/s.
The reference tiled implementation performs more small operations at narrow
windows, so speed gains depend on runtime overhead.

### Learnable attention sinks

Set `attention_sinks = true` to add one learned, zero-initialized sink
logit per head in every layer, including full-attention layers. It enters
the softmax denominator and has zero value contribution. The reference
backend supports this option in both PyTorch and MLX.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_mup_width256_rope_swa32_sinks_150m.toml
```

This fresh run matches the plain SWA-32 RoPE training settings and adds 16 sink
parameters across four layers and four heads. Zero initialization gives the
sink an initial score of zero and changes the normalization immediately.
Compare loss and throughput with `gpt-mlx-mup-width256-rope-swa32-150m-01`.

For a warmup–stable–decay comparison of the sink model:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_mup_width256_rope_swa32_sinks_wsd_150m.toml
```

It warms up for 250 updates, holds the same peak LR through update 5,860,
then cosine-decays to 10% of peak at update 7,325. The stable phase is set
by `lr_decay_start_step`. Total training targets and other settings match
the original cosine sink experiment.

### GPT-2 Small block dimensions with an 8k vocabulary

The larger config uses 12 layers, width 768, 12 heads and FFN width 3,072,
for 97,344,144 parameters. It retains context 256, RoPE, sinks, independent
input/output embeddings, and a repeating three-local/one-full layout.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_mup_gpt2_dims_150m.toml
```

Microbatch 4 and accumulation 20 preserve the effective batch of 80 sequences.
The 150.016M-target schedule reuses the smaller WSD experiment's LR.
Depth transfer of that LR remains to be measured. A two-update local probe
completed with 3.56 GiB peak MLX memory and about 8.8k tok/s on its second
update while another trainer was active. The full training duration requires
a longer measurement. Results are saved under
`artifacts/pretraining/gpt-mlx-mup-gpt2-dims-rope-swa32-sinks-wsd-150m-01`.

## Tiny mixture of experts (MLX)

Open [notebook 12](../../../notebooks/12_moe_tutorial.ipynb) for a routing
walkthrough, expert computations, balancing loss, and a completed tiny
memorization example with plots and source references.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_moe_tiny.toml
```

This uses a two-layer, width-64 RoPE model with four experts per layer, two
selected per token, and expert hidden width 128. It trains 256k targets over
1,000 updates and writes to `artifacts/pretraining/gpt-mlx-moe-tiny-01`.
The model has 677,504 parameters.

MoE config fields are `num_experts` (0 selects dense FFNs),
`experts_per_token`, `expert_hidden_dim` and `moe_aux_loss_coef`.
The first version supports MLX with standard parameterization. μP router
scaling remains a separate experiment.

Training minimizes cross-entropy plus the coefficient times mean layer
balancing loss. `train_loss` includes that term. Validation, fixed-probe losses,
and gradient-noise probes use cross-entropy. Each microbatch computes its own
balancing term. At statistics collection points, `probes.routing` records
per-layer assignment counts, mean probabilities, entropy and balancing loss on
the first fixed train probe.

Experts use bias-free two-matrix GELU networks. `gather_mm` executes selected
expert slots, and routing has no capacity limit or token dropping. With top-1,
the gate retains its full-softmax probability to preserve task gradients;
with multiple choices the selected gates are normalized to sum to one.
Benchmark actual throughput before treating increased sparsity as a speed gain.

## Tiny multi-head latent attention (MLX)

Open [notebook 13](../../../notebooks/13_mla_tutorial.ipynb) to follow the
projections, decoupled RoPE, expanded attention and compressed-cache computation.
Run a small corpus experiment using the existing tokenizer and token cache:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
  --config configs/pretraining/sequential_gpt_mlx_mla_width64_150m.toml
```

Results go to `artifacts/pretraining/gpt-mlx-mla-width64-150m-*`. This uses the
initial MoE run's 150M-token budget, context length and effective batch, with
warmup followed by stable LR. `sequential_gpt_mlx_mla_tiny.toml` is the shorter
1,000-update smoke config. The config selects
`attention_architecture = "mla"` and specifies query/KV latent ranks and
content, RoPE and value dimensions per head. It uses full causal attention,
standard parameterization, and dense FFNs. MoE FFNs can also be enabled.

`MultiHeadLatentAttention.decode(chunk, cache)` exposes compressed-cache
decoding at the attention-layer level. GPT generation currently recomputes its
cropped context. The tutorial verifies expanded and cached outputs agree.
This first version uses reference MLX operations and ordinary RoPE; μP,
sliding windows and sinks are unsupported for MLA.
