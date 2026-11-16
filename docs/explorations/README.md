# Explore later

## Fused kernels and sliding-window attention

Parked: MLX 0.32.4 fused backward accepts causal masking but rejects array masks;
our SWA uses an explicit causal window mask. Compilation alone does not replace
it with a fused attention kernel.

Checked 2026-11-16: [native window support PR #4734](https://github.com/ml-explore/mlx/pull/4734)
merged October 23. Current `main` has fused windowed forward, but explicitly
rejects windowed backward (`window support for backward attention is NYI`).
Latest stable remains 0.32.4. No open MLX PR implementing windowed backward
found in the current PR list/search. Nightly builds exist; current main still
has the training blocker.

Next: watch native windowed backward support before considering a custom Metal
implementation. Compare outputs, gradients, memory, and timing against our tiled
reference before training. [Current dispatch source](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp)
(search the error above).

Sources: [MLX mask guard](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp#L947-L952),
[custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).

Local checkout and probe are prepared: [setup instructions](../../experiments/mlx_swa/README.md).
Xcode and the Metal toolchain are installed; local build verification is in progress.
