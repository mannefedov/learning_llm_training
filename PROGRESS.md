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

## 2026-10-13 — Small GPT pretraining implementation

- **Changes:** Added a pure-PyTorch GPT-style model with explicit causal
  attention, learned positions, pre-normalized blocks, GELU, and tied weights.
  Added EOS document packing, shifted-target random batches, a single-device
  AdamW loop, validation loss, checkpoint saving, and a TOML-driven CLI using
  the existing FineWeb2 iterator and trained tokenizer. Included sampling from
  checkpoints in the root README; no knowledge docs were created.
- **Reference:** https://github.com/karpathy/build-nanogpt — architecture and
  training-loop inspiration. This does not load or reproduce GPT-2 checkpoints.
- **Verification:** Four tiny CPU tests passed: future-token masking, EOS and
  target alignment, loss reduction on a toy sequence, and a two-step local
  config run with checkpoint reload and generation. CLI help, TOML parsing,
  and diff whitespace checked. The local tokenizer has 8,000 tokens and EOS ID 0.
- **Left for the user:** Install `.[pretraining]`, run the FineWeb2 config,
  inspect train/validation losses, and sample text. No remote data or substantial
  training was run. Start with five updates before increasing the budget.
- **Current limits:** Full precision and in-memory token samples; constant
  learning rate; end-of-run checkpoint only; no automatic resume. Artifacts stay
  ignored and large-file storage is still deferred.

## 2026-10-13 — Publish initial pretraining implementation

- **User observation:** The user reported running training. Loss values and
  generated samples have not been reviewed in this session.
- **Changes being pushed:** GPT model, pretraining data and training functions,
  CLI, example TOML config, offline tests, and root README run instructions.
- **Verification:** The four tiny CPU tests passed during implementation;
  checked diff whitespace and confirmed `.env` and `artifacts/` are ignored,
  with no tracked artifacts.
- **Publishing:** User requested pushing to `origin/main`. Checkpoints and
  other artifacts remain local; large-file storage is still deferred.
- **Next hands-on step:** Inspect training and validation losses and sample
  from the saved checkpoint before choosing the next change.

## 2026-10-13 — Local run records

- **Changes:** Added `metrics.csv` for the existing step/train-loss/validation-loss
  output and `metadata.json` for timestamps, status, Git state, versions, resolved
  settings, device, parameter count, and tokenizer hash. Run folders now save
  config and tokenizer before data preparation; each CSV row is flushed so
  records survive Python exceptions or keyboard interruption.
- **Scope:** No new metrics, evaluation changes, plotting dependencies, or
  external tracking services. Evaluation design and comparison plots are deferred.
- **Verification:** Five small CPU tests passed, including CSV rows and metadata
  from a two-step local run, checkpoint reload, and preserving records on failure
  or interruption. Diff whitespace checked; no remote data or substantial training.
- **Left for the user:** Use a fresh output directory in each run config. New
  runs record their existing losses; previous checkpoints have no loss history.

## 2026-10-13 — Publish local run tracking

- **Changes being pushed:** CSV loss logging, JSON run metadata, runner
  integration, offline tests, and root README usage details.
- **Verification:** Five tiny CPU tests passed during implementation; checked
  diff whitespace and confirmed artifacts and `.env` remain ignored and untracked.
- **Publishing:** User requested pushing to `origin/main`. Run files stay local.
- **Next steps:** Decide evaluation metrics and design comparison plots.

<!-- Copy for future sessions:
## YYYY-MM-DD — Session topic

- **Focus:**
- **Changes / learning:**
- **Verification / observations:**
- **Left for the user:**
- **Open questions:**
- **Next hands-on step:**
-->

## 2026-11-13 — Parallel MLX pretraining on Mac

- **Changes:** Reviewed tokenizer, model, data, training, tracking, CLI, configs,
  tests, and prior project notes. Added `models/gpt_mlx.py` and
  `pretraining_mlx/` without modifying the existing PyTorch implementation.
  Preserved causal attention, learned positions, pre-norm blocks, tanh GELU,
  tied weights, initialization scales, shifted EOS-packed batches, validation,
  matrix-only AdamW decay with bias correction, clipping, and metrics cadence.
  Shared the existing tokenizer, dataset iterator, and CSV/status writer.
- **Artifacts:** MLX saves model and optimizer Safetensors with a JSON checkpoint
  manifest, config, tokenizer, metadata, and CSV losses. Added a separate
  five-step MLX config, optional dependency extra, run and sampling commands.
  No checkpoint conversion or automatic resume; seeded batches differ between
  backends because MLX batching uses an independent Python RNG.
- **References:** https://ml-explore.github.io/mlx/build/html/python/optimizers.html
  and https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GELU.html
  for MLX optimizer state, bias correction, and tanh GELU API behavior.
- **Verification:** Created ignored `.venv` with Python 3.12; installed MLX
  0.32.4 and PyTorch 2.15.0 for checks. All 18 offline tests passed, including
  existing tests, MLX toy loss reduction, two-step config/checkpoint reload and
  generation, deterministic validation, decay filtering, and one Metal GPU
  update. Identical-weight logits/loss and two clipped optimizer updates agreed
  with PyTorch within test tolerances. CLI help and diff whitespace passed.
  MLX runtime checks required running outside the sandbox for Metal access.
- **Left for the user:** No remote corpus fetch or substantial training was run.
  Activate `.venv` and run `python -m llm_training.pretraining_mlx --config
  configs/pretraining/tiny_gpt_mlx.toml` after ensuring the tokenizer exists.
  Inspect finite train/validation losses and sample the saved model; increase
  the step budget only after the small run, with a fresh output directory.
- **Next steps:** Compare actual runs before unifying config validation, ID
  packing, and run setup. Keep model math and checkpoint handling explicit for
  each backend until shared behavior is established. No commit or push requested.

## 2026-11-13 — Investigate tokenizer shutdown hang

- **Observation:** User reported the tokenizer prints its final saved message
  but does not exit. Both output files are written before that message. The
  original tokenizer process was no longer running when inspected; did not
  interrupt the user's active MLX pretraining process.
- **Investigation:** Found matching upstream partial-Parquet-stream shutdown
  reports: https://github.com/huggingface/datasets/issues/7467 and
  https://github.com/apache/arrow/issues/45214. This is a likely explanation,
  not a confirmed stack diagnosis of the user's stopped process.
- **Changes:** Added explicit bounded-stream iterator closure and garbage
  collection while Python is still running, including early-close and error
  paths. Shared data loading means this also applies to both pretraining paths.
  Did not force process exit, change dependencies, or change corpus sampling.
- **Verification:** Offline test added for completion, early closure, and invalid
  text cleanup. Full offline suite passed (19 tests). No remote fetch or full
  tokenizer retraining; the original shutdown hang has not been reproduced or
  confirmed resolved. Existing saved tokenizer remains usable.
- **Next step:** Observe normal shutdown on the next intentional remote run.
  If the hang persists, inspect that live process's native stack before choosing
  a dependency workaround. Use a new output directory for a fresh tokenizer run.

## 2026-11-13 — Automatic run directories

- **Changes:** Added a shared exclusive directory allocator to tokenizer,
  PyTorch pretraining, and MLX pretraining runners. Repeated paths increment
  trailing numbers (`run-01` to `run-02`); unnumbered names gain `-02`.
  Existing files/directories are skipped, preserving prior artifacts. Runners
  print and return the actual paths; metadata uses the allocated run name.
  Updated README and config guidance. Original requested config copies remain
  intact; this does not add checkpoint resume.
- **Verification:** All 21 offline tests passed, including repeated config runs
  in all three runners, existing-file collisions, and number-width overflow.
  Diff whitespace passed. No remote fetching or substantial training.
- **Left for the user / next step:** Rerun the same command after changing steps;
  inspect the printed run directory. Each invocation starts a new model. Select
  the newly printed tokenizer path explicitly when using a newly trained tokenizer.

## 2026-11-13 — Commit Mac training implementation

- **User observation:** User reports MLX training works; no loss curves or
  generated samples reviewed in this session.
- **Changes committed:** Parallel MLX model and pretraining pipeline, dependency
  extra, config, checkpoints and sampling instructions; automatic output run
  numbering in all runners; explicit dataset stream cleanup and offline tests.
  Preserved user's MLX config settings (5,000 steps, context 256, ten validation
  batches) and aligned README smoke-run guidance.
- **Verification:** Latest full offline suite passed all 21 tests, including
  one Metal update and PyTorch/MLX numerical comparison. Final diff whitespace
  passed; artifacts, virtual environments, and `.env` remain ignored.
- **Left for the user:** Substantial training and evaluation; stream-shutdown
  mitigation remains unconfirmed against the original reported hang.
- **Next steps:** Inspect real run losses and samples before backend unification.
- **Publishing:** User requested a local commit; no push requested.

## 2026-11-13 — Publish MLX implementation and inspect run records

- **Publishing:** User requested pushing the committed MLX implementation,
  automatic run directories, tokenizer cleanup, and tests to `origin/main`.
- **Verification:** Prior full offline suite passed 21 tests; working tree was
  clean before this log update. Artifacts and credentials remain ignored.
- **Local observations:** Latest completed run `tiny-gpt-mlx-run-03` reached
  5,000 steps. Recorded validation loss decreased from 9.0061 to 5.3375, with
  its lowest recorded value at step 5,000. Training loss on that final batch
  was 4.9940; this is not an interval mean. No new training or generation run.
- **Next step / left for user:** Inspect loss curves and checkpoint samples
  before increasing training budget or changing architecture. Existing runs
  have different settings, so final losses alone do not isolate a cause.

## 2026-11-13 — Checkpoint generation exploration

- **Changes:** Added `python -m llm_training.pretraining_mlx.generate` with run
  selection, repeatable prompts/seeds, multiple temperatures, token limit,
  device selection, and EOS stopping. Loads saved model/tokenizer together;
  no optimizer or dataset loading. Added reusable checkpoint/text generation
  functions and `notebooks/02_mlx_generation.ipynb` for manual exploration.
  README includes the command and links the notebook.
- **Verification:** Ten tiny offline MLX tests passed, including saved checkpoint
  loading, seed repeatability, EOS stopping, and invalid sampling arguments.
  CLI help, notebook code-cell compilation, and diff whitespace passed.
- **Left for the user:** Run the command or notebook against the existing trained
  run and inspect continuation quality across prompts, temperatures, and seeds.
  No real-checkpoint quality exploration, new training, or remote fetch performed.
- **Next step:** Compare repetition, sentence structure, and prompt consistency;
  use observations together with validation losses to choose the next change.

## 2026-11-13 — Periodic generation during MLX training

- **Changes:** Added optional `[generation]` settings, enabled by default. MLX
  config runs append readable continuations to `samples.txt` at step zero,
  every configured interval (default 500), and the final step. Default prompt
  is `Сегодня утром я`; example config supplies two Russian prompts. Each entry
  records step, prompt, temperature, seed, and token limit and is flushed.
  Disabled generation creates no sample file. Metadata records resolved defaults.
- **Implementation:** Reused standalone generation with local MLX PRNG keys;
  model generation accepts an optional explicit key. The callback restores model
  training mode even on error and does not change training or global RNG state.
- **Verification:** All 23 offline tests passed, including sample cadence,
  enabled/disabled config runs yielding identical model outputs, EOS stopping,
  RNG isolation, mode restoration on failure, and flushed sample retention.
  Syntax and whitespace checks passed. No substantial training or remote fetch.
- **Left for the user / next step:** Run the normal MLX training command and
  inspect or follow `<run-directory>/samples.txt`. Adjust `generation.every`,
  prompts, temperature, seed, and token limit; set `enabled = false` to skip it.
  Existing checkpoints and historical runs are unchanged.

## 2026-11-13 — Compact training progress

- **Changes:** Added a dependency-free progress display shared by PyTorch and
  MLX loops, replacing repeated evaluation print lines. Shows percentage,
  completed/total steps, elapsed time, remaining-time estimate, and latest
  logged losses. Refreshes a single terminal line at most once per second;
  redirected logs get initial/final status and at most one line per 30 seconds.
  Narrow terminals omit the visual bar and truncate to avoid wrapping.
- **Timing:** ETA uses average wall time per completed update, including
  validation and sampling. Data preparation and checkpoint writing are excluded.
  Early estimates may be noisy. Interruptions/errors print stopped status at
  actual recorded progress rather than claiming completion.
- **Verification:** All 26 offline tests passed, covering deterministic elapsed
  time/ETA/percentage, sparse redirected output, interruption cleanup, and
  existing training/sampling/checkpoint checks. Diff whitespace passed.
- **Left for the user / next step:** Run the usual training command to inspect
  the live display on a longer run. No substantial training, remote fetching,
  or new dependencies. CSV metrics and sample-file cadence remain unchanged.

## 2026-11-13 — Compare loss curves across runs

- **Changes:** Added `notebooks/03_compare_run_losses.ipynb` with run selection,
  separate training/validation curves, optional rolling training-loss smoothing,
  initial-step filtering, latest/best validation summaries, and settings/tokenizer
  hashes for comparison context. Rerunning the final cell reloads running CSVs;
  incomplete final lines are excluded. Added optional `analysis` Matplotlib
  dependency and README entry. No training code or run artifacts changed.
- **Verification:** Parsed every code cell; checked partial CSV rows, empty
  files, metadata rewrite fallback, and smoothing on tiny local fixtures.
  Executed the notebook cells against six existing runs, including the active
  run, with isolated Matplotlib dependencies under `/private/tmp`. Rendered and
  visually inspected both loss plots; checked smoothed/zoomed refresh. Notebook
  remains unexecuted on disk, without saved outputs. Diff whitespace passed.
- **Left for the user / next step:** Install `.[analysis]` in the notebook kernel
  environment, select runs, and rerun the final cell for updated curves.
  Runs 05/06 have a different tokenizer hash from earlier runs, so compare settings
  before attributing loss differences to training budget. No new model training,
  remote data access, or changes to the active training environment.

## 2026-11-13 — Notebook prose preference

- **Preference:** Keep notebook prose minimal and focused on running and
  understanding code; avoid unsolicited caveats and generic comparison advice.
  Saved in `AGENTS.md` for future sessions.
- **Changes:** Removed the interpretation/caveat cell from the loss comparison
  notebook. No code changes. Notebook JSON parsed and diff whitespace checked.

## 2026-11-13 — Short, runnable human-facing instructions

- **Preference:** Human-facing READMEs/instructions should be short, readable,
  and command-oriented; detailed agent guidance and continuity records belong
  in `AGENTS.md` and `PROGRESS.md`. Saved this distinction in `AGENTS.md`.
- **Changes:** Replaced the long root README with a five-step Mac workflow and
  stage links. Added module READMEs for tokenization, MLX pretraining, PyTorch
  pretraining, models, and post-training status. Simplified notebook instructions
  and replaced stale docs indexes with links to runnable module instructions.
- **Verification:** Checked all 11 READMEs, 29 local links, shell/Python snippet
  syntax, and diff whitespace. Documentation-only changes; no training run.
- **Next step:** User review before committing. Existing pending code, notebooks,
  and user config edits remain intact; no commit or push performed this turn.

## 2026-11-13 — Publish generation, progress, and notebooks

- **Changes being pushed:** Standalone and periodic MLX generation, sample files,
  compact progress for both training backends, generation/loss notebooks,
  plotting extra, concise workflow/module READMEs, and writing preferences.
  Preserved user's current configs: 10,000-document tokenizer/training samples,
  the second tokenizer artifact path, 200,000 steps, batch size 10, and selected
  generation prompts. Artifact paths refer to local files, not committed assets.
- **Verification:** Latest offline suite passed all 26 tests. Loss notebook
  rendered against existing runs with isolated plotting dependencies; partial
  row and refresh checks passed. README links/snippet syntax and final diff
  whitespace checked. No new substantial training or model evaluation.
- **Left for user / next steps:** Continue the active run, inspect samples, and
  refresh loss plots. Existing training artifacts and `.env` remain untracked.
- **Publishing:** User requested committing and pushing to `origin/main`.
