# Tokenizer training

Run from the repository root, with your Python environment activated.

```sh
python -m pip install -e '.[tokenization]'
python -m llm_training.tokenization --config configs/tokenization/fineweb2_bpe.toml
```

Edit `[data]` for the corpus and sample size; edit `[tokenizer]` for vocabulary
size and BPE settings. Put `HF_TOKEN=...` in `.env` if authentication is needed.

The command prints the output directory containing `tokenizer.json` and
`config.toml`. Repeated runs automatically choose a new numbered directory.

Next: set `[tokenizer].path` in your [MLX](../pretraining_mlx/README.md) or
[PyTorch](../pretraining/README.md) config to that tokenizer file.

For hands-on inspection, open [the BPE notebook](../../../notebooks/01_bpe_fineweb2.ipynb).
