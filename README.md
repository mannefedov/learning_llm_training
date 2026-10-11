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

Each Python module should point to its relevant Markdown note in its module
docstring. Keep source links, explanations, and paper notes in `docs/`; keep
comments about how the implementation works next to the code.

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

