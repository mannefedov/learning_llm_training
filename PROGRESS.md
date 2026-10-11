# Progress

Append one entry per meaningful session. Record what changed, what you learned
or still need to understand, checks performed, and the next hands-on step.
Update this log before pushing work.

## 2026-10-11 — Repository scaffold

- **Focus:** Establish a minimal structure for learning LLM training.
- **Changes:** Created packages for tokenization, models, pretraining, and
  post-training; added matching documentation, a reference-note template,
  a notebooks directory, and agent guidance.
- **Decisions:** Keep fine-tuning and RL under post-training initially. Add
  dependencies and experiment tracking when concrete implementations need them.
- **Verification:** Checked Python package imports, metadata parsing, local
  documentation links, and whitespace. No training or experiments were run.
- **Open questions:** Which first component will make a useful starting point?
- **Next hands-on step:** Choose a small tokenizer or model component, collect
  its sources, and explore it in a notebook.
- **Publishing:** The user requested committing and pushing this scaffold to
  `origin/main`. Substantial training and notebook exploration remain for the user.

## 2026-10-11 — BPE library and corpus research

- **Focus:** Find an existing trainable BPE implementation and check FineWeb2 access.
- **Changes / learning:** Recorded Hugging Face Tokenizers as the recommendation,
  a stable version to start with, and a streaming byte-level BPE exercise in
  `docs/tokenization/bpe_fineweb2.md`. FineWeb2 uses non-English language-script
  subsets; English data is in the original FineWeb.
- **Verification / observations:** Checked official repository sources and
  dataset documentation; syntax-checked the example and checked diff whitespace.
- **Left for the user:** Dependency installation, corpus streaming, tokenizer
  training, and inspection of tokenization behavior. No dataset or training runs.
- **Open questions:** Which language or language mix should the first tokenizer use?
- **Next hands-on step:** Run the example on 100 documents, then increase the
  sample and compare vocabulary sizes on held-out text.

## 2026-10-11 — Tokenization exploration notebook

- **Changes:** Moved the BPE example into `notebooks/01_bpe_fineweb2.ipynb`;
  removed the premature suggestion and exercise document from `docs/`.
- **Preference recorded:** Documentation records knowledge, not suggestions or
  exercises. Create or expand knowledge docs only on direct user request.
  Saved this guidance in `AGENTS.md` and aligned the root README.
- **Verification:** Checked notebook structure, parsed Python cells, and checked
  diff whitespace. Installation, data fetching, and training were not run.
- **Left for the user:** Run the notebook and inspect tokens, vocabulary,
  merges, and held-out token counts.

## 2026-10-11 — Runnable tokenizer training

- **Changes:** Added `train_byte_level_bpe` for reusable training, a bounded
  Hugging Face text iterator, and `python -m llm_training.tokenization --config`.
  Added a top-level TOML config and optional tokenization dependencies. Runs save
  a tokenizer JSON and config copy under ignored `artifacts/`, rejecting existing
  output directories. Added usage commands to the root README; no knowledge docs.
- **Verification:** Four tiny offline tests passed, covering unseen-character
  round trips and serialization, invalid input, document limits and blank
  filtering, and config-driven saving with overwrite prevention. CLI help,
  example config parsing, package metadata parsing, and diff whitespace passed.
  The local dataset tests required permission for Hugging Face cache lock files.
- **Left for the user:** Install the package extra and run the 100-document
  FineWeb2 config. No remote dataset fetching or substantial training was run.
- **Next hands-on step:** Inspect the saved tokenizer, then change sample size
  or vocabulary size in a copied config with a new output directory.

## 2026-10-11 — Script environment file

- **Changes:** Added an ignored local `.env` with an empty `HF_TOKEN`, a tracked
  `.env.example` template, and `python-dotenv` to the tokenization extra. The
  training runner loads `.env` from the working directory before importing
  Hugging Face libraries, preserving variables already exported in the shell.
- **Verification:** Checked loading with dummy values, existing-variable
  precedence, Git ignoring `.env`, and diff whitespace. No network or training.
- **Left for the user:** Fill in the local token and install the updated extra.

## 2026-10-11 — Publish tokenization setup

- **Changes being pushed:** BPE training functions and CLI, TOML config,
  exploration notebook, offline tests, environment template, and updated agent
  guidance. Preserved the user's small notebook outputs.
- **Verification:** Prior offline checks passed; checked diff whitespace and
  confirmed `.env` and `artifacts/` are ignored, with no tracked artifacts.
- **Publishing:** User requested pushing to `origin/main` without artifacts.
  Large-file storage decisions are deferred to a future session.

<!-- Copy for future sessions:
## YYYY-MM-DD — Session topic

- **Focus:**
- **Changes / learning:**
- **Verification / observations:**
- **Left for the user:**
- **Open questions:**
- **Next hands-on step:**
-->
