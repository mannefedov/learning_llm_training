#!/bin/bash
# Run from the repository root. Compare each result with the completed 1k anneal.
set -euo pipefail

source_run=artifacts/pretraining/gpt-mlx-mup-width256-2b-03
trainer=.local/venv-mlx-swa/bin/python
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export PYTHONPATH=src

for factor in 2 4 8; do
  updates=$((1000 / factor))
  config="configs/pretraining/sequential_gpt_mlx_mup_anneal_batch${factor}x.toml"
  echo "Batch ${factor}x: ${updates} updates, 20.48M training targets"
  "$trainer" -m llm_training.pretraining_mlx \
    --config "$config" --resume "$source_run" --resume-new-cache \
    --anneal-steps "$updates" --allow-accumulation-change
done
