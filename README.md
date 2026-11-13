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
output paths use the current working directory. Existing run paths are preserved;
a numbered directory is created automatically on repeated runs. Pin the optional dataset revision for repeatable data selection.

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

Outputs go under ignored `artifacts/pretraining/`. Set a descriptive base
directory in the TOML config:

```text
<run directory>/
    config.toml
    metadata.json
    metrics.csv
    tokenizer.json
    checkpoint.pt
```

`metadata.json` records timestamps, run status, Git commit and modified-code
flag, library versions, resolved model/training settings, device, parameter
count, and tokenizer hash. `metrics.csv` records only the existing console
values: `step`, `train_loss`, and `val_loss`. Rows follow the existing logging
cadence; training loss is the current batch loss, not an interval average.
Step zero has an empty training-loss field. Evaluation settings are unchanged.

Config, tokenizer, and metadata are saved before data preparation; CSV rows
are flushed during training. Failed or interrupted runs retain their records.
The checkpoint includes model and optimizer states, dimensions, config, and
step. It is saved at the end; automatic resume and periodic saving are not
implemented yet. Earlier runs cannot recover loss history from checkpoints.

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


## Run the parallel MLX implementation on Apple silicon

The original PyTorch modules and config remain available. The MLX path uses
`models/gpt_mlx.py` and `pretraining_mlx/`, sharing the existing Hugging Face
BPE tokenizer, dataset iterator, and CSV/status writer. It does not require
PyTorch. Use an Apple silicon Mac and a Python version supported by MLX;
Python 3.12 was used for local verification.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pretraining-mlx]'
python -m llm_training.pretraining_mlx --config configs/pretraining/tiny_gpt_mlx.toml
```

Train the tokenizer with the existing tokenization command first if its JSON
is not already present. The MLX config currently uses 5,000 updates, context
length 256, and ten validation batches. For a first smoke run, set `steps = 5`
and `eval_batches = 1`. Existing output paths are skipped automatically. `device = "auto"` selects the Metal GPU when
available; explicit choices are `"gpu"` and `"cpu"`.

Architecture, initialization scales, float32 training, matrix-only weight
decay, AdamW betas and bias correction, clipping, and logging cadence match
the PyTorch path. Sampling windows use a separate Python RNG, so the same seed
does not produce identical batches or initial weights across frameworks.
MLX explicitly evaluates each update to keep its lazy computation graph bounded.
See the [MLX optimizer reference](https://ml-explore.github.io/mlx/build/html/python/optimizers.html).

Run directories contain the same config, tokenizer, metadata, and metrics files,
plus `model.safetensors`, `optimizer.safetensors`, and `checkpoint.json` (the
manifest with dimensions, config, and step). Optimizer state can be restored
with `optimizer.state = tree_unflatten(mx.load(path))` using `mlx.utils`.
Automatic resume and conversion of PyTorch checkpoints are not implemented.
Each invocation starts from random weights.

Sample an MLX checkpoint:

```python
import json
from pathlib import Path
import mlx.core as mx
from tokenizers import Tokenizer
from llm_training.models.gpt_mlx import GPT, GPTConfig

run_dir = Path("artifacts/pretraining/tiny-gpt-mlx-run-01")
checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
model = GPT(GPTConfig(**checkpoint["model_config"]))
model.load_weights(str(run_dir / checkpoint["model"]))
model.eval()
tokenizer = Tokenizer.from_file(str(run_dir / "tokenizer.json"))
prompt = mx.array([tokenizer.encode("Привет", add_special_tokens=False).ids])
generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(generated[0].tolist()))
```

Run only the tiny offline MLX checks:

```sh
python -m unittest discover -s tests -p 'test_pretraining_mlx.py' -v
```

The model and training configs intentionally remain parallel for now. After
using both paths, shared config validation, document ID packing, and run setup
are candidates for unification; model math, autodiff, and checkpoint handling
can stay explicit per backend. No backend abstraction is required yet.

All three runners automatically allocate a fresh directory if the configured
path exists: `tiny-gpt-mlx-run-01` becomes `tiny-gpt-mlx-run-02`, then `03`,
and so on. Names without a trailing number gain `-02`, `-03`, etc. The actual
path is printed and returned by `train_from_config`; previous artifacts are
preserved. Saved configs retain the original requested path, while pretraining
metadata records the actual run name. Each training invocation still starts a
new model; increasing steps does not resume the previous checkpoint.
