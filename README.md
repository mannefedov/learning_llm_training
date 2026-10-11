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

