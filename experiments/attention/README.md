# Attention analysis

Start the local viewer from the repository root. Requires the analysis extra
and saved attention probes under `artifacts/analysis/attention-sinks`.

```sh
.venv/bin/python experiments/attention/serve_viewer.py
```

Open http://127.0.0.1:8765/ and leave the terminal running. Use `--port` to
select another port, or `--data` to choose a probe directory.

- Wheel zooms around the pointer. Drag pans the map. Touch supports pinch zoom.
- The overview shows the visible rectangle. Click it to center a region.
- Fit or double-click restores the whole map. Full screen expands the workspace.
- With the canvas focused, arrow keys pan, +/− zoom and F fits.
- Hover reads the exact probability, with masked future cells labeled separately.

The canvas uses the current saved arrays and draws only the selected head.
Layer/head changes keep the camera position. Axes remain visible while panning.
Refresh probes reloads the available checkpoints and their maps.

To collect fresh probes on the MLX machine:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer PYTHONPATH=src \
  .local/venv-mlx-swa/bin/python experiments/attention/inspect_sinks.py
```

The probe script currently lists three checkpoint paths explicitly. Edit its
directory list to choose models. Run it after training has saved a checkpoint.
Notebook 11 contains the reading guides and static overview plots.

Check camera navigation without a browser:

```sh
node --test experiments/attention/viewer/camera.test.mjs
```
