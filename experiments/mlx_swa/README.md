# Local MLX SWA development

Prerequisite: install full **Xcode** (including the Metal toolchain), launch it
once to finish setup, then verify `xcrun metal --version` succeeds. If needed, select it:

```sh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
```

From the training repo root, build in the isolated environment:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
source .local/venv-mlx-swa/bin/activate
cd .local/mlx-swa
python -m pip install -e '.[dev]'
cd ../..
python experiments/mlx_swa/probe.py --expect-backward-unavailable
```

The probe checks fused windowed forward and confirms the current backward error.
After implementing backward, run it without the flag to compare Q/K/V gradients.
It starts with head size 64, float32, causal windows, and small synthetic inputs.

Rebuild after changing MLX:

```sh
cd .local/mlx-swa
python setup.py build_ext --inplace
```

Checkout: `.local/mlx-swa`, branch `experiment/windowed-backward`.
Starting commit: `fc344c986c4169d99f76b5d95ae6b4a58a359a63`.
The checkout and environment are ignored by the training repo; keep kernel
changes in that checkout's branch. The training repo's `.venv` is separate.

Start in `mlx/backend/metal/scaled_dot_product_attention.cpp`: search for
`window support for backward attention is NYI` and `SDPAVJPTileParams`.
Trace window parameters through backward dispatch and tile computations before
removing the guard. Extend `python/tests/test_fast_sdpa.py` for upstream tests.

[MLX build instructions](https://ml-explore.github.io/mlx/build/html/install.html#build-from-source).
