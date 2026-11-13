# Learning LLM Training

Small implementations for learning how LLMs are trained.

## Run the workflow on Mac

Run all commands from the repository root. Use Python 3.12 on Apple silicon.

**1. Set up the environment**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pretraining-mlx,analysis]'
```

Add `HF_TOKEN=...` to your local `.env` if your dataset needs authentication.

**2. Train a tokenizer**

```sh
python -m llm_training.tokenization --config configs/tokenization/fineweb2_bpe.toml
```

**3. Train a GPT model**

Set `[tokenizer].path` in the training config to the `tokenizer.json` printed
by step 2. For a quick check, set `steps = 5` and `eval_batches = 1`.

```sh
python -m llm_training.pretraining_mlx --config configs/pretraining/tiny_gpt_mlx.toml
```

The terminal shows progress, losses, elapsed time, and ETA. Generated samples
are saved periodically to `samples.txt` in the printed run directory.

**4. Generate text**

Use the completed run directory printed by step 3:

```sh
python -m llm_training.pretraining_mlx.generate \
  --run-dir artifacts/pretraining/tiny-gpt-mlx-run-01 \
  --prompt "Сегодня утром я" --temperature 0.8 --max-new-tokens 100
```

**5. Plot the losses**

Open [the loss notebook](notebooks/03_compare_run_losses.ipynb) in your notebook
editor and select the `.venv` kernel. Run its cells; rerun the final cell to refresh
while training continues.

## Run individual stages

| Stage | Instructions |
| --- | --- |
| Tokenizer | [Train and reuse BPE](src/llm_training/tokenization/README.md) |
| Mac / MLX | [Train, sample, and configure generation](src/llm_training/pretraining_mlx/README.md) |
| PyTorch | [Train on CUDA, MPS, or CPU](src/llm_training/pretraining/README.md) |
| Models | [GPT implementations](src/llm_training/models/README.md) |
| Notebooks | [Tokenization, generation, and loss plots](notebooks/README.md) |
| Post-training | [Current status](src/llm_training/post_training/README.md) |

Configs live in `configs/`; outputs stay under ignored `artifacts/`.
Existing output directories get a new number automatically. Each training
invocation starts a new model; checkpoint resume is not implemented.

## Checks

With the relevant backend extras installed:

```sh
python -m unittest discover -s tests -v
```

Agent guidance lives in [AGENTS.md](AGENTS.md); session history is in
[PROGRESS.md](PROGRESS.md).
