#!/bin/bash
set -euo pipefail

export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export PYTHONPATH=src
for steps in 1000 3000 5000; do
  .local/venv-mlx-swa/bin/python -m llm_training.pretraining_mlx \
    --config "configs/pretraining/sequential_gpt_mlx_mup_staircase_${steps}.toml" \
    --resume artifacts/pretraining/gpt-mlx-mup-width256-2b-03 \
    --resume-new-cache --anneal-steps "$steps" --anneal-schedule staircase
done
