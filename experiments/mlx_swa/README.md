# Local MLX attention experiments

Parked: NAX backward showed no noticeable gain in the user's training run.
The existing tiled backward remains the default. Keep this code for learning
and later profiling; no long-run quality comparison has been performed.

## Build

Requires full Xcode with the Metal toolchain, a local MLX checkout at
`fc344c986c4169d99f76b5d95ae6b4a58a359a63`, and an isolated Python environment.
Checkout: `.local/mlx-swa`; environment: `.local/venv-mlx-swa`.
[Upstream build instructions](https://ml-explore.github.io/mlx/build/html/install.html#build-from-source).

For a fresh checkout, apply these patches in order (skip already applied patches):

```sh
git -C .local/mlx-swa apply ../../experiments/mlx_swa/causal_window_backward.patch
git -C .local/mlx-swa apply ../../experiments/mlx_swa/flash64_backward.patch
git -C .local/mlx-swa apply ../../experiments/mlx_swa/flash64_nax_backward.patch
source .local/venv-mlx-swa/bin/activate
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd .local/mlx-swa
python setup.py build_ext --inplace
cd ../..
```

## Select backward

Set environment variables before starting Python:

| Mode | Variables | Requirements |
| --- | --- | --- |
| Existing tiled (default) | `MLX_SDPA_VJP_FLASH64=0` | Native fused attention |
| On-chip SIMD prototype | `MLX_SDPA_VJP_FLASH64=1 MLX_ENABLE_TF32=0` | FP32, D64, equal heads/lengths, causal |
| M5 NAX prototype | `MLX_SDPA_VJP_FLASH64=2 MLX_ENABLE_TF32=1 MLX_SDPA_VJP_FLASH64_TILE=16` | Same shape restrictions, NAX hardware |

NAX tile choices16/32/64 mean Q×K tiles16×32/32×32/32×64. Both prototypes
support full causal and window attention. They use separate dQ/dKV ownership
and recompute scores; NAX keeps score intermediates in registers. Mode1 requires
TF32 off to match forward precision. Mode2 uses native NAX arithmetic.

The window patch supports equal-length causal training with sequence length>8;
array masks and sinks remain unsupported. Reference backend remains available
in model config. Use the local environment for `mlx_fused_window` configs.

## Verify and benchmark

From the repo root, with the local environment active:

```sh
python experiments/mlx_swa/check_backward.py
python experiments/mlx_swa/test_extended.py
MLX_ENABLE_TF32=0 python experiments/mlx_swa/flash_backward.py --suite
MLX_ENABLE_TF32=1 python experiments/mlx_swa/flash_backward.py --nax-grid
python experiments/mlx_swa/flash_backward.py --confirm-performance
```

`--confirm-performance` compares fresh temporary models at batches2/8 using a
repeated cached batch. It records strict multi-update comparison failures for
diagnosis; initial gradient checks still raise. Native repeat controls also
failed parameter equality. Attention-gradient comparisons passed; matching
short loss trajectories do not establish long-run quality.

Additional tools: `profile_training.py` for component timing, `audit_step.py`
for precision-preserving optimizations, `sweep.py` for longer contexts, and
`allocation_limit.py` for the dense score-buffer limit. Sweeps are explicit,
optional experiments; the host-RSS watchdog is not a GPU-memory cap.

Raw benchmark output belongs under `artifacts/benchmarks/`. Intermediate reports
and results from this session remain locally in
`.local/mlx-swa-reports-2026-11-16/` and are excluded from Git.

## Resume exploration

First reproduce a benefit in matched real training runs; prior microbenchmarks
suggested gains that were not noticeable in the user's run. Then inspect GPU
counters for register spills/occupancy and test cooperative tile reuse/pipelining.

Kernel sources: `mlx/backend/metal/scaled_dot_product_attention.cpp` and
`mlx/backend/metal/kernels/scaled_dot_product_attention_vjp.metal`. NAX helpers:
`mlx/backend/metal/kernels/steel/attn/nax.h`.

References: [FlashAttention](https://arxiv.org/abs/2205.14135),
[MLX tiled backward](https://github.com/ml-explore/mlx/pull/4563),
[Apple TensorOps](https://developer.apple.com/videos/play/tech-talks/111432/).
