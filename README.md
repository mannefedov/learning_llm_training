# Learning LLM Training

A personal learning project for understanding LLM training and preparing for interviews.
Build small, readable implementations, explore the ideas in notebooks, and keep
notes that connect code to its sources.

## Layout

```text
src/llm_training/
    tokenization/     # Text representation and tokenizer training
    models/           # Model architectures and components
    pretraining/      # Data preparation, objectives, and training loops
    post_training/    # Fine-tuning, preference optimization, and RL
docs/
    tokenization/
    models/
    pretraining/
    post_training/
    note_template.md
notebooks/            # Freeform explorations
PROGRESS.md           # Session log
AGENTS.md             # Guidance for future coding sessions
```

Link Python modules to relevant existing knowledge notes in their docstrings.
Create knowledge notes in `docs/` only on explicit request; put suggestions and
exercises in notebooks. Keep implementation comments next to the code.

## Getting started

The scaffold has no runtime dependencies. Add dependencies as the first
implementation needs them.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Use `from llm_training import ...` in scripts and notebooks after installing the
package in the environment used by your notebook kernel.

## Run tokenizer training

From the repository root:

Put your Hugging Face token in the local `.env` file as `HF_TOKEN=hf_...`.
The training runner loads this file without replacing environment variables
already set in your shell. `.env` is ignored by Git; `.env.example` provides
the empty template for new checkouts.

```sh
python -m pip install -e '.[tokenization]'
python -m llm_training.tokenization --config configs/tokenization/fineweb2_bpe.toml
```

Edit the TOML file to change the dataset subset, sample size, and BPE settings.
The example streams 100 Russian FineWeb2 documents. Configs live in
`configs/tokenization/`; learned tokenizers go into the ignored `artifacts/`
directory. Each run saves `tokenizer.json` and a copy of its config. Relative
output paths use the current working directory; choose a fresh output directory
for each run. Pin the optional dataset revision for repeatable data selection.

Use the same training function in notebooks:

```python
from llm_training.tokenization.bpe import train_byte_level_bpe

tokenizer = train_byte_level_bpe(texts, vocab_size=8_000)
```

Small offline verification:

```sh
python -m unittest discover -s tests -v
```

## Run small GPT pretraining

The model uses explicit causal multi-head attention, learned positions,
pre-normalized residual blocks, GELU, and tied embedding/output weights. The
model and optimizer are implemented in PyTorch, with
[Karpathy's build-nanogpt](https://github.com/karpathy/build-nanogpt) as the
architecture reference. This is a small GPT-style model trained from random
weights, not a reproduction of GPT-2's pretrained weights.

From the repo root, with your trained tokenizer already available:

```sh
python -m pip install -e '.[pretraining]'
python -m llm_training.pretraining --config configs/pretraining/tiny_gpt.toml
```

The config selects a small FineWeb2 training sample and a separate `test` sample
used for validation. Keep any final test set separate when tuning against this
validation sample. Documents are concatenated with EOS separators and random
context windows are sampled with targets shifted by one token. The entire
tokenized sample stays in CPU memory. EOS marks document boundaries but does
not block attention across them.

The default is four layers, width 128, context 128, batch size 8, and 500
optimizer steps. Set `steps = 5` and `eval_batches = 1` for your first smoke run.
It uses full precision, constant learning rate, AdamW, and gradient clipping.
Training loss and validation loss are printed; a tiny run is for checking the
pipeline, not for producing fluent text. Each invocation starts a new model.

Outputs go under ignored `artifacts/pretraining/`: `checkpoint.pt` (model and
optimizer states, dimensions, config, and step), `tokenizer.json`, and
`config.toml`. Choose a fresh output directory for each run. Checkpoints are
saved at the end; automatic resume and periodic saving are not implemented yet.

To sample from a saved checkpoint in a notebook or script:

```python
import torch
from tokenizers import Tokenizer
from llm_training.models.gpt import GPT, GPTConfig

run_dir = "artifacts/pretraining/tiny-gpt-run-01"
checkpoint = torch.load(f"{run_dir}/checkpoint.pt", map_location="cpu", weights_only=True)
model = GPT(GPTConfig(**checkpoint["model_config"]))
model.load_state_dict(checkpoint["model"])
model.eval()
tokenizer = Tokenizer.from_file(f"{run_dir}/tokenizer.json")
prompt = torch.tensor([tokenizer.encode("Привет", add_special_tokens=False).ids])
generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(generated[0].tolist()))
```

