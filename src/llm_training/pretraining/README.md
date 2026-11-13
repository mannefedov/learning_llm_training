# PyTorch pretraining

Run from the repository root, with your environment activated.
[Train a tokenizer](../tokenization/README.md) first and set `[tokenizer].path`
in `configs/pretraining/tiny_gpt.toml` to its saved JSON.

```sh
python -m pip install -e '.[pretraining]'
python -m llm_training.pretraining --config configs/pretraining/tiny_gpt.toml
```

Edit `[model]` and `[training]` in the config. A quick check uses `steps = 5`
and `eval_batches = 1`. `device = "auto"` selects CUDA, then MPS, then CPU.

The printed run directory contains config/tokenizer copies, `metrics.csv`,
`metadata.json`, and the final `checkpoint.pt`. Repeated runs get new numbered
directories and start from random weights.

## Generate from a checkpoint

In a Python session or notebook, use your completed run path:

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

For loss plots, install `.[analysis]` and open
[the comparison notebook](../../../notebooks/03_compare_run_losses.ipynb).
Periodic sample files and the generation CLI currently belong to the
[MLX path](../pretraining_mlx/README.md).
