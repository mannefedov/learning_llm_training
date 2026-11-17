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

## 2026-11-13 — Fixed-budget sequential disk corpus

- **Decision:** Keep 200,000 optimizer steps, batch 10, context 256. Prepare only
  the required 512,000,001 token IDs (512 million targets), with 20 million
  document records as an upper limit. Documents are streamed in source order;
  the final document may be truncated. Do not fetch 20 million documents merely
  to leave most unused, and do not random-resample training windows.
- **Changes:** Added a disk preparation CLI and separate sequential MLX config.
  Writes little-endian uint32 train/validation token files and a completion
  manifest; memory-maps them read-only and converts only batches to MLX arrays.
  Sequential batching visits each target position once without wrapping.
  Existing random/in-memory training remains available. Validation uses the
  existing fixed random windows, drawn from a separate cached split.
- **Cache behavior:** Reuses complete compatible caches without fetching data.
  Checks tokenizer hash/version, source settings, and file sizes; records actual
  documents/tokens and final-document truncation. Incomplete or undersized caches
  fail explicitly. No preparation resume; choose a new cache directory after an
  interruption. Metadata records cache identity/counts. Declared NumPy directly
  in the MLX extra (already installed transitively).
- **Verification:** All 30 offline tests passed, including stopping at budget,
  stream cleanup, cache reuse/mismatch/corruption, incomplete-source failure,
  exact sequential target coverage, memmap validation batches, and two-step
  config training with remote fetching forbidden. CLI help, budget arithmetic,
  syntax, README links, and diff whitespace passed.
- **Left for user:** Run prepare and training commands from the MLX README with
  `configs/pretraining/sequential_gpt_mlx.toml`. The train token file will be
  2,048,000,004 bytes (~1.91 GiB), plus validation. No large data preparation,
  downloads, or substantial training were run; user's small config edits remain
  untouched. Inspect printed document/token totals and subsequent loss curves.

## 2026-11-13 — Confirmed Arrow shutdown deadlock after cache preparation

- **Observed:** Inspected the live preparation process without stopping it. A
  one-second native stack sample showed the main thread in C runtime exit,
  waiting in `arrow::internal::ThreadPool::Shutdown` from libarrow 26. Python
  finalization had already completed; earlier explicit iterator cleanup did not
  prevent this native shutdown deadlock. Stack saved locally under `/private/tmp`.
- **Cache verified:** Completion manifest exists. Train file has 512,000,001
  tokens from 432,767 documents (last truncated); validation has 99,372 tokens
  from 100 documents. Both file sizes exactly match uint32 token counts. No
  full-file checksum/read or retraining performed.
- **Changes:** Preparation CLI now prints `Cache ready`, flushes stdout/stderr,
  and calls `os._exit(0)` after successful main completion, bypassing problematic
  native finalizers. Imported preparation functions retain normal behavior;
  exceptions do not enter the success exit path. Scope limited to this CLI.
  Reference: https://github.com/apache/arrow/issues/45214.
- **Verification:** Real existing-cache preparation command exited successfully
  without downloads. Subprocess test confirmed success skips shutdown hooks,
  while preparation errors retain nonzero exit, traceback, and normal hooks.
  Diff whitespace checked. A fresh remote streaming preparation was not repeated.
- **Left for user:** Stop the previously stuck preparation process and start
  sequential training with the existing complete cache. No rebuild is needed.

## 2026-11-13 — Publish sequential disk-cache training

- **Changes being pushed:** Budget-limited corpus preparation, read-only memmap
  cache reuse, sequential MLX batching, separate experiment config, metadata,
  short run instructions, and success-only CLI native-shutdown workaround.
  Preserved user's small-config change to 200,000 training document records.
- **Verification:** Full offline suite passed 30 tests before the CLI workaround;
  the additional subprocess test passed afterward. Existing-cache preparation
  exited successfully with no download. Final diff whitespace passed. User
  reports the updated workflow works; no new training outcomes inferred.
- **Left for user / next step:** Continue sequential training and inspect losses
  and generated samples. No fresh large preparation or substantial training run
  was performed by the agent. Caches, checkpoints, and credentials stay ignored.
- **Publishing:** User requested committing and pushing to `origin/main`.

## 2026-11-13 — Modular full and sliding-window attention

- **Changes:** Extracted backend-specific attention modules and a shared GPT
  config. Each block uses a small explicit factory; supported per-layer types
  are `full` and `sliding_window`. No plugin registry or backend framework.
  Missing attention fields preserve full attention for old configs/checkpoints;
  projection parameter paths, shapes, initialization, and counts are unchanged.
- **Algorithm:** Window W includes self, so query i sees max(0,i-W+1)..i.
  SWA computes query tiles of W positions against at most 2W-1 keys with a causal
  band mask. This bounds score storage/computation by O(TW) rather than masking
  a full T-by-T score matrix. Standard MHA and learned positions remain; no
  KV cache or custom kernels. No measured speed/memory benefit claimed.
- **Research:** https://arxiv.org/abs/2408.00118 (Gemma 2) and
  https://arxiv.org/html/2503.19786v1#S5.SS2 (Gemma 3). Reported layouts are
  1:1 local/global with window 4096 in Gemma 2, and 5:1 with window 1024 in
  Gemma 3. Gemma 3's text-only ablations report little perplexity impact from
  several ratios. These are reported large-model results, not local findings.
- **Experiment choice:** Separate four-layer 1:1 and 3:1 sequential MLX configs,
  local window 64 at context 256. Start with 1:1 as a simple four-layer mix;
  compare 3:1 without changing depth. Window 64 is an experimental choice,
  not an established optimum. Both configs reuse the existing token cache and
  retain baseline data, model width/depth, and training budget. Active run and
  baseline configs were not modified.
- **Verification:** All 35 offline tests passed. SWA outputs and gradients match
  an independent dense band-mask reference for windows 1, 3, T, and >T;
  boundaries, causal exclusion, tile sizes, selection, config validation,
  full-window equivalence, and old config defaults checked. PyTorch/MLX logits
  and two optimizer updates agree for full, local, and mixed models. Tiny mixed
  Metal update passed. Reran 13 MLX tests after adding SWA checkpoint reload and
  generation coverage; all passed. Config budget equivalence, notebook syntax,
  rendered mask plots, README links, and diff whitespace checked.
- **Left for user / next step:** Run `python -m llm_training.pretraining_mlx
  --config configs/pretraining/sequential_gpt_mlx_swa_1to1.toml`, then compare
  the 3:1 config and baseline in notebook 03. Notebook 04 contains source links
  and mask exploration. No substantial training, sweeps, or speed benchmark run.
  Open question: which window and layer mix best preserve quality here?

## 2026-11-13 — Publish attention variants

- **Changes being pushed:** Modular full/SWA attention in both backends, shared
  GPT config, 1:1 and 3:1 experiment configs, mask notebook, short model README,
  and attention tests. Existing defaults and baseline configs remain intact.
- **Verification:** Full offline suite passed 35 tests; subsequent 13-test MLX
  run passed with SWA checkpoint reload/generation coverage. Notebook rendering,
  matched experiment settings, local links, and final diff whitespace checked.
- **Left for user / next step:** Continue active training and compare full, 1:1,
  and 3:1 loss curves and samples. No substantial training or performance
  benchmark performed by the agent. Training process and local artifacts untouched.
- **Publishing:** User requested committing and pushing to `origin/main`.

## 2026-11-13 — Bounded MLX checkpoints and resume

- **Changes:** Added `[checkpoint] every=1000, keep=1` defaults to MLX runs.
  Saves completed step directories with weights, optimizer, config, host batch
  RNG state, tokenizer hash, and cache identity; the run's checkpoint.json points
  to the latest complete checkpoint. Publishes new files before pruning older
  owned step directories. No duplicate final model/optimizer files at run root.
  `every=0` saves only at the end. Retention is per run, with one extra temporary
  copy during saves; interrupted Python saves clean their temporary files.
- **Resume:** `--resume` accepts a run directory or retained checkpoint.json.
  Restores optimizer/model state, completed step/data position, and batch RNG;
  validates model/data/tokenizer and optimizer settings before creating output.
  Steps is the total target; resumed logs go into a new numbered run with source
  recorded in metadata. ETA counts only newly completed steps. Earlier-format
  checkpoints remain usable for generation but cannot resume through this CLI.
- **Verification:** All 38 offline tests passed. Interrupted/resumed mixed-SWA
  models and optimizer tensors matched uninterrupted CPU runs exactly for both
  random and sequential sampling. Checked retention 1/2, final-only saving,
  preserved previous checkpoint after simulated disk-full failure, retained
  checkpoint generation, incompatible resume rejection, and resumed ETA.
  CLI help, configs, README links/snippets, and diff whitespace passed.
- **User follow-up:** Latest local 3:1 checkpoint is format 2 at step 200,000,
  already the configured target. No further steps remain in this token budget.
- **Left for user:** New runs use periodic checkpoints; use --resume after an
  interruption with the original config and a target above the saved step.
  No substantial training or downloads performed by the agent; active process,
  user notebook edits, and unrelated config edits preserved. Not committed yet.

## 2026-11-13 — Sequential epochs beyond one corpus pass

- **Changes:** Sequential batches map the global step modulo full batches per
  epoch, repeating ordered passes until the total step target. Added shared
  epoch-size validation, startup/metadata epoch counts, and fractional epochs
  in progress output. Steps remains the only budget setting; no extra epoch
  config is required. A sub-batch tail is omitted on each pass (the existing
  512-million-token budget divides exactly, so it has no unused target tail).
- **Resume:** Completed or interrupted version-2 checkpoints resume in the
  correct epoch and batch using the saved global step, preserving optimizer
  state. The same cache can now support a higher total step count.
- **Preparation:** Sequential preparation accepts document/source limits below
  the requested token budget and reuses complete smaller caches for multiple
  epochs. Non-sequential preparation retains the prior short-source error.
  Existing datasets/caches were not modified; no large downloads performed.
- **Verification:** All 38 offline tests passed, including wraparound targets,
  invalid/small corpus checks, a completed one-pass checkpoint resumed into
  epoch two, and interrupted/resumed CPU models and optimizers matching an
  uninterrupted run across epoch boundaries. Smaller cache preparation/reuse,
  epoch progress, and diff whitespace checked.
- **Left for user / next step:** Set training.steps to 400000 in the 3:1 config,
  keep the current cache, and resume gpt-mlx-swa-3to1-512m-01 for a second pass.
  No substantial training started by the agent. Unrelated notebook/config edits
  preserved. Checkpointing and epoch changes remain uncommitted.

## 2026-11-13 — Publish checkpointing, resume, and epochs

- **Changes being pushed:** Periodic MLX checkpoints with keep=1 default,
  retention-safe publication, resume validation/restoration, multi-epoch
  sequential batching/cache reuse, epoch progress, tests, and short instructions.
  Includes user's 400,000-step 3:1 config, small-config document count change,
  and existing loss notebook outputs without modifying them.
- **Verification:** Latest full offline suite passed all 38 tests, including
  exact interrupted/resumed model/optimizer equivalence across epochs, retention,
  simulated save failure, and smaller cache reuse. Notebook JSON/code syntax and
  final diff whitespace checked; training artifacts and credentials excluded.
- **Left for user / next step:** Continue the second epoch and inspect losses
  and samples. No substantial training/downloads run by the agent; active process
  unaffected. PyTorch resume is outside this MLX change.
- **Publishing:** User requested committing and pushing to `origin/main`.

## 2026-11-14 — Audit recent softmax numerical report

- **Research:** Barron, Gnanapragasam, and Tschannen, A Fix for Softmax and
  Cross-Entropy (submitted October 22, revised October 29, 2026):
  https://arxiv.org/abs/2610.31642 and https://jonbarron.info/softmaxfix/.
  Distinguishes hard-target softmax collapse from soft-target cancellation.
  Author page includes an erratum for NaN gradients in an earlier proposed fix
  when masked rows have one effective finite logit; do not copy that early fix.
- **Local verification:** Tiny float32 probes with logits [9,-9,-9], target 0,
  reproduced collapse on PyTorch 2.15.0 CPU/MPS and MLX 0.32.4 CPU/Metal.
  Cross-entropy loss and target gradient were zero; two non-target gradients
  remained about 1.523e-8 each, giving gradient sum about 3.046e-8. Analytic loss
  is 3.046e-8, target gradient -3.046e-8, with zero gradient sum. Probed actual
  shared attention helpers too: dominant-probability gradients similarly lost
  their maximum entry. Both full and SWA use those helpers. MLX's single-valid-key
  masked row retained finite gradients with the existing library implementation.
- **Scope:** Current hard-target loss and attention are susceptible under these
  numerical conditions. Soft-target distillation cancellation is outside the
  implemented training objective. Did not measure occurrence/severity on trained
  checkpoints or infer damage to prior experiments. No model training, downloads,
  dependency updates, production math changes, or active-process changes.
- **Reproduction:** Temporary probe script/output remained under /private/tmp;
  no speculative knowledge docs/notebook added. Next possible task is a focused
  corrected-math implementation with masked-row/gradient regression checks.

## 2026-11-14 — Opt-in corrected softmax training experiment

- **Changes:** Added backend-specific stable log-softmax/softmax helpers using
  an explicit pivot, sum of non-pivot exponentials, and log1p. Hard-target CE
  gathers from the corrected log-softmax, keeping the implementation small.
  Config `[model] stable_softmax=true` selects the corrected attention and loss
  in both backends, plus PyTorch sampling. MLX categorical generation already
  operates directly on logits. Defaults remain false for baseline/old configs.
- **Source and differences:** https://jonbarron.info/softmaxfix/ section 2 and
  masked-row erratum. Uses log1p directly, not log(sum) followed by softplus;
  the latter can differentiate through log(0). MLX argmax indices explicitly
  stop gradients, while the gathered pivot value remains differentiable.
  This implements the correction needed for current attention/hard-target
  training, not a new soft-target distillation objective. Rows require at least
  one finite logit; existing causal/SWA masks keep self visible.
- **Experiment:** Added sequential_gpt_mlx_swa_3to1_stable.toml matching the
  current 3:1 config except stable_softmax and output path. Reuses the existing
  cache. Checkpoint config records the flag; changing math during resume is
  rejected, so use a fresh run for the comparison.
- **Verification:** All 44 offline tests passed on the final run. Confident
  float32 loss/target gradients remain nonzero on CPU, MPS, and Metal; actual
  attention-helper gradients checked. Single finite/effectively finite rows,
  one-key windows, ties, large/wrong-target logits, ordinary float64-reference
  agreement, Torch double gradcheck, and 8k-class MLX reductions checked.
  The 8k float32 reduction retains small rounding error (tested with relative
  tolerances), rather than catastrophic loss of the dominant gradient.
  Backend comparisons cover both flags across full/local/mixed attention.
  Corrected-math checkpoint/resume/retention tests and changed-math rejection
  passed. Experiment settings equivalence, README links, and whitespace checked.
- **Left for user / next step:** Run the stable config from scratch and compare
  curves/samples to the baseline. No substantial training, downloads, or speed
  benchmark run, and no active-process changes or dependency upgrades. Existing
  experiments and configs were preserved. Not committed or pushed.

## 2026-11-14 — Research MLX custom-kernel optimization

- **User observation:** User reports similar/worse generated examples and slower
  corrected-math training after 400k steps; no controlled quality/speed analysis
  performed in this session.
- **Research:** Official MLX 0.32.4 guides for custom Metal kernels,
  custom_function/VJP, and compile:
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html
  https://ml-explore.github.io/mlx/build/html/usage/compile.html
- **Findings:** Python mx.fast.metal_kernel accepts Metal Shading Language source
  with explicit launch grid/threadgroups and output shapes/dtypes. Training needs
  a registered backward/VJP, commonly with a second custom kernel. mx.compile
  can fuse some ordinary operations first. Safe math matters for masked -inf
  and corrected reductions. Local standard MLX loss dispatches to fast.cross_entropy
  on Metal; our correction is composed array operations, so extra kernels and
  temporaries are plausible causes of slowdown, not a profiled attribution.
- **Scope / next step:** Explained the kernel model and proposed first optimizing
  corrected CE forward/backward, benchmarked against the reference and library.
  No kernels implemented, benchmarks run, production changes, or new docs made;
  user asked for general explanation only.

## 2026-11-14 — Compile MLX forward/backward

- **Changes:** Added opt-in `[training] compile_step=true` (default false).
  Compiles the complete loss/value-and-gradient function once, with model.state
  explicit as implicit inputs/outputs so optimizer updates cannot leave stale
  captured weights. Optimizer updates, validation, generation, logging, and
  checkpoint writes stay outside. Compilation is a runtime setting that may
  change on resume, unlike the stable_softmax model setting.
- **Experiment:** Added a compiled variant of the corrected 3:1 config, differing
  only in compile_step and output path. Notebook 05 benchmarks native, corrected,
  forward-only compilation, forward+backward compilation, and the full model.
  Uses synthetic tensors and synchronized evaluations; no data/checkpoints loaded.
- **Local measurements:** Forward-only compilation of corrected helpers showed
  negligible benefit (~1.0x). Compiling isolated CE+gradient together reduced one
  short measurement from ~12.9 to ~9.9 ms. Full four-layer model forward/backward
  at batch 10/context 256/vocab 8000, after warmup and five measured evaluations:
  native 11.183 ms, corrected 25.204 ms, compiled corrected 18.762 ms (~25.6% less
  time than corrected, ~1.34x throughput). This final repeat ran after our tests
  finished. Timings exclude optimizer, validation, generation, checkpointing,
  and first-use compilation; not a full-training throughput/quality result.
- **Verification:** All 45 offline tests passed. Checked compiled/uncompiled
  confident and masked-row gradients, updates with changing weights, tiny mixed
  models, checkpoint retention/resume, and toggling compilation during resume.
  Config equivalence, notebook code syntax, README links, and whitespace passed.
- **Left for user / next step:** Run the compiled config (or set compile_step in
  the existing config and resume normally) and compare observed progress timing.
  Notebook 05 reproduces isolated timings. No substantial training, custom GPU
  kernels, new dependencies, or active-process changes. Changes uncommitted.

## 2026-11-14 — Attention efficiency and model optimization audit

- **Audit:** Full attention computes the complete score matrix and masks it
  afterward. Existing SWA computes only nearby query/key tiles, giving O(TW)
  score work, but still materializes tile scores/weights and runs separate
  operations. Compilation does not substitute a specialized attention kernel.
  Packed QKV, batched heads, tied embeddings, and mmap batches are already used.
- **Research:** FlashAttention-2 uses tiling, online softmax/output accumulation,
  on-chip storage, skipped wholly future blocks, and masking at boundaries.
  Dense causal attention still has O(T²) arithmetic; local attention can restrict
  work to O(TW). Source: https://arxiv.org/html/2307.08691v1 . Current MLX API:
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
  and version-pinned https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/fast.cpp .
- **Local probe:** Float32 [1,4,256,32] inputs on Metal accepted causal and band
  masks with default fallback and finite forward/backward results. With
  force_fused=True, causal attention failed because head dim 32 is unsupported
  (message also notes float32/TF32 restrictions); band-masked attention failed
  because fused backward accepts causal masking only, not an array mask.
  Therefore the native fast API is not a verified fused drop-in for this setup.
  Installed Python API has no native window_size keyword. Changing head layout
  or precision needs a separate experiment; do not silently change architecture.
- **Other candidates:** Output logits at batch 10/context 256/vocab 8000 occupy
  ~78.1 MiB; one full layer's score matrix occupies 10 MiB. Corrected CE currently
  materializes full log probabilities and is a strong optimization candidate.
  Also consider measured optimizer/synchronization overhead, mixed precision as
  a separate test, and generation's full-prefix recomputation/all-position output
  projection. Learned positions require care when caching a sliding context.
- **Scope:** Research/code inspection and a tiny synthetic compatibility probe
  only. No production changes, new training, architecture changes, or speed
  benchmark in this audit. Recommend preserving reference attention, profiling
  components, then adding explicitly supported fused paths and corrected CE
  kernels rather than expecting a causal mask alone to avoid score computation.

## 2026-11-16 — Exploration backlog and fused full-attention experiment

- Added docs/explorations/README.md for explicitly parked topics; AGENTS.md
  records this cross-session convention as an exception to completed-work docs.
  Parked fused SWA: MLX 0.32.4 array-mask backward restriction, source, and next
  investigation (native window support or custom Metal kernels).
- Earlier speedup was compilation, not fused attention. Added opt-in model
  attention_backend=mlx_fused (reference remains default), using causal string
  mask and force_fused=True. Unsupported GPU/shapes fail rather than fallback;
  corrected softmax and SWA combinations rejected, PyTorch entry point rejects
  this MLX-only setting. Checkpoint model equality includes the new field.
- New sequential_gpt_mlx_full_fused.toml uses all-full attention, two heads of
  size 64 (previous four heads had unsupported size 32), native softmax, GPU,
  and compiled forward/backward. Existing token cache reused; new run required.
  For kernel-only comparisons set backend=reference with same other settings.
- Verification: 48 offline tests passed on Metal. Added output/QKV gradient
  comparisons at lengths 16/33/256, short-prefix inference at 1/7, compiled tiny
  model backward and generation, incompatible config rejection. Initial tight
  tolerances exposed float32 fused/reference differences: fixed-seed synthetic
  probes showed output max ~0.00204 and gradient max ~0.00189; tests allow
  atol/rtol 0.003. Short prefixes <=8 support fused inference but not backward
  in installed version; training config uses 256. TOML and git diff --check pass.
- Left for user: run new config and inspect loss, samples, and throughput.
  No substantial training or speed benchmark run; active training untouched.

## 2026-11-16 — Upstream fused SWA research

- Queried public GitHub API for all open MLX PRs and window/backward PR search;
  no open windowed-backward implementation found. Relevant native window PR
  #4734 merged 2026-10-23, after installed/latest stable v0.32.4. Current main
  supports window_size for forward but has explicit output_logsumexp + window
  rejection: window support for backward attention is NYI. Array mask restriction
  also remains. Updated exploration backlog with findings and source links.
- Release workflow schedules daily wheels/artifacts (scheduled publication is
  disabled by its publish expression); did not install builds or change venv.
  Main source itself still blocks fused windowed training, regardless of build.
- PR #4563 (merged Sep 29) explains Metal backward performance tradeoff: a
  specialized recomputing kernel did not improve speed; tiled Steel matmuls
  provided memory savings and speed. This supports implementation complexity as
  a likely reason for incremental coverage, not a confirmed maintainer reason
  for SWA specifically. No substantial computations or active-run changes.

## 2026-11-16 — Local SWA development setup

- Cloned upstream main at fc344c986c4169d99f76b5d95ae6b4a58a359a63 into
  ignored .local/mlx-swa; created experiment/windowed-backward branch.
  Read upstream AGENTS.md; no kernel edits or upstream submissions made.
- Created isolated .local/venv-mlx-swa and installed CMake, setuptools,
  typing_extensions, numpy, pytest, pre-commit. Training .venv unchanged.
- Added experiments/mlx_swa README and synthetic forward/QKV-gradient probe.
  Expected-failure flag checks the specific upstream window backward rejection;
  default mode requires implemented gradients. Covers self-only, partial tiles,
  window larger than sequence, and 256/64. Syntax and diff checks passed.
- BLOCKER: full Xcode is absent from /Applications; active developer path is
  CommandLineTools and xcrun cannot find metal. Build and GPU probe remain unrun.
  User must install/initialize Xcode and Metal toolchain, then follow README
  build/probe commands. Next: verify pristine build and reproduce guard, then
  implement window propagation and tile bounds in backward.

## 2026-11-16 — Save iteration before local kernel work

- Commit includes corrected softmax/CE, compiled MLX forward/backward, forced
  fused full-attention config, numerical/attention tests and benchmark notebook,
  exploration backlog convention, and isolated SWA development instructions/probe.
- Verification carried forward: 48 offline tests passed on Metal after model
  changes; SWA probe syntax and whitespace checks passed. Full training outcomes
  are user-run, not agent-verified.
- Xcode 27.2 and Metal Toolchain 27B47 now verified installed; metal --version
  works outside sandbox with DEVELOPER_DIR set. Updated build instructions.
- Isolated editable MLX build started in exec session 18709 and is still running
  at this checkpoint. Checkout fc344c9, local experiment/windowed-backward branch,
  .local/venv-mlx-swa. These ignored local assets are not included in this commit.
- Next iteration: inspect build result, run expected-failure SWA probe, then
  implement/test windowed backward. No upstream kernel edits yet.


### 2026-11-16 — MLX window backward, throughput metrics, and parked kernel experiments

- Added `mlx_fused_window` model backend and 4K 3local/1full training config.
  Preserved user's batch8/15625-step budget; corrected its explanatory comment.
  Window backend requires the isolated patched MLX environment. Reference and
  full-only fused backends remain available.
- Added cumulative elapsed/tokens/tok-per-second reporting to Torch and MLX
  training progress and CSV metrics. Notebook parser accepts both old and new
  CSV columns. Throughput includes in-loop reporting/evaluation/sampling costs.
- Local MLX starts at fc344c986c4169d99f76b5d95ae6b4a58a359a63. Saved three ordered
  patches: causal_window_backward, flash64_backward, flash64_nax_backward.
  First extends tiled backward with causal windows; second adds FP32/D64 on-chip
  SIMD kernels; third adds M5 TensorOps/register tiles and three tile shapes.
  Exact patch-chain reconstruction verified against current local source.
  Mode0(default)=tiled; mode1=SIMD/TF32off; mode2=NAX/TF32on. NAX16 means16×32
  Q/K tile. Other experimental restrictions: equal heads/lengths, causal,D64.
- Benchmarks located most time in forward/backward; optimizer compilation/data
  loading did not offer substantial gains. Smaller tiled backward blocks helped
  small batches. Original SIMD prototype saved memory but was slower and needed
  TF32off for consistent recomputation. NAX restored native TF32 compatibility.
  Short paired benchmarks suggested ~3–4% B2/~11% B8 gains over tuned tiled256,
  but user observed no noticeable gain in actual training. Parked without changing
  defaults. Resume with matched real-run timing and GPU counters before further
  tile reuse/pipelining work. BF16 and long-document continuation remain deferred.
- Verification before parking:60 strict attention comparisons across three NAX
  tile shapes; initial model-gradient differences <=7.57e-6; shape/precision guards,
  eager/compiled and window isolation passed. Default regression17tests and two
  experimental window tests passed. Original SIMD FP32 smoke still passed.
  Multi-update parameter equality failed, including unchanged-native repeats;
  small loss differences do not establish long-run quality. Diagnostic suites
  retain failed-array reporting; strict initial-gradient checks still raise.
- Kept reproducible scripts/tests/patches and concise experiment README. Raw results,
  intermediate reports and detailed session log archived locally under ignored
  `.local/mlx-swa-reports-2026-11-16/`; excluded from this commit at user's request.
  No long training/quality run performed by agent. Current user training left alone.
  Final commit checks: focused CPU test suite, Python syntax, notebook syntax and
  Git whitespace checks (outcome recorded below). User requested commit, no push.
- Final verification:29 focused tests ran (28passed,1expected skip for the custom
  MLX API in the stock environment). Experiment scripts and notebook code cells
  parsed/compiled; git diff --check passed. GPU tests were not rerun while the
  user's training is active; results above were obtained before that run.

## 2026-11-16 — Optimization review of saved training runs

- Read MLX/Torch optimizer loops and saved configs, metadata, checkpoints, and
  loss CSVs; no training or evaluation executed and no training behavior changed.
- Small 1.85M SWA model: validation ~9.009 at initialization, ~4.712 at 200k,
  ~4.691 at 400k. Larger 6.26M/4K run: ~9.031 to ~3.848 over 62.5k steps.
  Both one-pass budgets process 512M tokens; different model/context/eval windows
  mean their losses do not isolate the effect of model capacity.
- Current optimizer: matrix-only AdamW, betas (0.9,0.95), clip1, constant3e-4.
  Recommended next controlled experiment: linear warmup then cosine decay to
  10% of peak; first keep peak3e-4, other settings, and token budget fixed.
  Warmup is for fresh training, not a remedy for an existing late plateau.
  Scheduler implementation and user-run comparison remain future work.
- Run full-fused-01 config copy says3e-3, but metadata and checkpoint both say
  3e-4; do not treat it as evidence of an actual higher-LR experiment.
- Reference checked: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
  uses linear warmup and cosine decay with a 10%-of-peak floor.

## 2026-11-16 — Lower-LR checkpoint continuation

- User requested a runnable lower-LR continuation. Selected completed 6.26M/4K
  SWA run gpt-mlx-swa-fused-4k-512m-01: checkpoint step62,500, validation~3.848.
  Added sequential_gpt_mlx_swa_fused_4k_low_lr.toml from its saved config:
  batch2 unchanged, constant LR3e-5, total68,750 steps (6,250 extra updates,
  51.2M targets). Fresh output base gpt-mlx-swa-fused-4k-low-lr-01.
- Resume now allows learning_rate changes and applies configured LR after loading
  optimizer state, retaining Adam moments/step and other compatibility checks.
  Updated MLX instructions. No scheduler or warmup added.
- Verified:13 MLX pretraining CPU tests passed, including exact restored weights,
  moments/step preservation and a lower-LR resumed update/checkpoint. Config model,
  tokenizer/data/cache identity, manifest and immutable training settings checked
  against source checkpoint; model/optimizer files exist. Whitespace check passed.
- Full continuation and GPU quality evaluation left to user. Command requires
  existing patched .local/venv-mlx-swa and --resume of the selected source run;
  compare held-out loss/sample quality to source. No active training modified.
- One isolated test invocation aborted during MLX import with duplicate nanobind
  DeviceType registration; immediate full-suite rerun passed all13 tests (1.38s).

## 2026-11-16 — Extend successful lower-LR continuation

- Inspected completed user-run gpt-mlx-swa-fused-4k-low-lr-01: steps62,500–68,750,
  val loss3.84849→3.67891; best at final step. Mean val over63,750–65,000 was
  3.71725 versus3.68684 over67,500–68,750, supporting further continuation.
  Observed runtime359.8s for6,250 updates. No new agent training performed.
- Added sequential_gpt_mlx_swa_fused_4k_low_lr_long.toml: total125,000 steps,
  constant3e-5 and batch2 unchanged, output gpt-mlx-swa-fused-4k-low-lr-long-01.
  Resume from low-lr-01 step68,750 adds56,250 updates/460.8M targets, reaching
  two total passes/1.024B targets; estimated54min at observed run speed.
- Verified source checkpoint model/optimizer files exist and config model/data/
  tokenizer/cache and all training fields except total steps match checkpoint.
  Whitespace check passed. User to run continuation and inspect validation trend
  and samples; plateau time remains unmeasured. No kernel/trainer changes.

## 2026-11-16 — MLX linear warmup and cosine LR decay

- Added opt-in lr_schedule="cosine", warmup_steps, min_learning_rate and optional
  lr_decay_steps to MLX TrainConfig. Constant remains default for existing runs.
  One-based update1 uses peak/warmup_steps; warmup ends at peak; cosine reaches
  floor on absolute lr_decay_steps (default training.steps), then stays there.
  No warmup starts at peak on update1. Invalid schedule/boundary settings reject.
- Trainer sets LR before each optimizer update. Absolute step determines LR on
  resume, retaining moments/count and avoiding warmup restart. Explicit decay
  horizon preserves curve when total budget is extended; schedule edits deliberately
  replan curve. Existing checkpoint LR override support retained.
- Added learning_rate CSV column (blank for initial/no-update row and Torch runs)
  and lr in MLX progress output. Updated stage README with settings/resume semantics.
  Reference: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py;
  local implementation uses explicit one-based peak/floor endpoints.
- New sequential_gpt_mlx_swa_fused_4k_cosine.toml is a fresh run using same4K model/
  batch2/cache, 125k total updates, 1250warmup, peak3e-4/floor3e-5. Output base
  gpt-mlx-swa-fused-4k-cosine-01. Existing active lower-LR continuation/configs
  untouched; no new training launched.
- Verification:27 focused CPU tests passed across MLX/Torch pretraining, checkpoint,
  and progress. New checks cover boundaries, validation, used/logged LR, serialized
  optimizer resume vs uninterrupted updates; existing compiled checkpoint test now
  also covers cosine and exact state equivalence. All MLX configs parsed/validated;
  sample schedule values2.4e-7(update1),3e-4(1250),1.65e-4(63125),3e-5(125000+).
  Standalone MLX config-validation import failed due sandbox Metal access; same
  lightweight command succeeded with escalation. Whitespace check passed.
- User to run fresh scheduled experiment and inspect validation/sample quality.
  No long GPU run or convergence result claimed.

## 2026-11-16 — Refresh training loss between evaluations

- User noticed train display changed only on eval. MLX and Torch loops now pass
  the latest batch loss to progress on every update; MLX also refreshes current LR.
  Cache scalar loss once after optimizer update and reuse for logging/display.
  Existing terminal throttle (1s interactive/30s redirected), validation and CSV
  cadence unchanged. Added concise MLX README explanation.
- Verification:25 focused pretraining/progress tests passed; git diff --check
  passed. No long training executed; running processes retain already loaded code.

## 2026-11-16 — Compare published LR ranges and schedules

- User asked whether3e-4→3e-5 is too narrow and how modern recipes compare.
  Verified primary sources: nanoGPT uses6e-4→6e-5 cosine after2000warmup;
  Llama3 405B uses8e-5→8e-7 cosine after8000warmup, then final linear annealing
  to0; DeepSeekV3 warms to2.2e-4 over2k updates, holds until10T tokens, cosine
  to2.2e-5 over4.3T, then final rates2.2e-5/7.3e-6; SmolLM3 uses2e-4 WSD,
  2000warmup and linear-to-zero decay over final10% updates.
- Ratio10x is a defensible baseline, not a tuned optimum for this project.
  Separate absolute peak, floor ratio, shape/horizon. Adam LR is not a fixed
  fraction of parameter values. Local cosine config's warmup starts2.4e-7.
  Suggested isolated comparisons: peak6e-4 with10%floor, or unchanged peak3e-4
  with1%floor3e-6; keep data/model/batch/budget fixed. No experiments run/changed.
- MiniCPM reports nominalLR0.01 with muP and parameter-dependent scaling;
  raw LR numbers across parameterizations cannot be copied directly.
- Sources: https://github.com/karpathy/nanoGPT/blob/master/train.py ;
  https://arxiv.org/html/2407.21783v3#S3.SS4 ;
  https://arxiv.org/html/2412.19437v2#S4.SS2 ;
  https://huggingface.co/blog/smollm3 ; https://arxiv.org/html/2404.06395v2

## 2026-11-17 — Explore training diagnostics before optimizer changes

- User wants established monitoring/gradient visualization/replay approaches
  before changing accumulation or optimization. Reviewed MLX loop: clip_grad_norm
  already returns global norm, currently discarded; logging has losses/LR/throughput.
  No trainer changes, app implementation, or computations performed.
- Discussed diagnostic priorities: preclip norm/clipping frequency; per-tensor
  gradient RMS and actual Adam update RMS/weight RMS; fixed train/held-out probe
  losses and token-level errors; optional activation statistics on fixed inputs;
  at fixed weights, multiple-batch gradient disagreement/noise estimates to assess
  accumulation. These suggest hypotheses, not definitive capacity diagnoses.
- Proposed modest replay UI: step slider over layer×time heatmaps, tensor details,
  fixed-example predictions, sampled metrics and sparse checkpoints. Individual
  weight movies/2D loss landscapes are secondary; projections cannot prove trapping.
  MLX lazy arrays require materializing preupdate snapshots if measuring true deltas;
  activation probes should be separate from compiled production forward/backward.
- Established references checked: TensorBoard scalar/histogram/embedding logging
  https://docs.pytorch.org/docs/stable/tensorboard ; TorchExplorer PyTorch module
  activations/parameters/gradient UI https://github.com/spfrommer/torchexplorer ;
  W&B watch source docs https://github.com/wandb/gitbook/blob/master/library/integrations/pytorch.md
  (live docs URL failed fetch); gradient noise scale https://arxiv.org/abs/1812.06162 ;
  normalized loss-landscape visualization https://arxiv.org/abs/1712.09913 ;
  Karpathy educational activations/gradient material https://github.com/karpathy/nn-zero-to-hero .
- No installation or external uploads. Further instrumentation awaits scope choice;
  existing training remains untouched. No claim about cause of latest plateau.

## 2026-11-17 — Optional MLX training-statistics collection and basic-model config

- User authorized statistics instrumentation before visualization/accumulation,
  returning to an earlier short-context model and parking attention optimizations.
  Added opt-in [statistics] every/probe_batches (default disabled), recorded in
  metadata. statistics.jsonl flushed per report, schema_version1, at initial/resume,
  update1, configured intervals, final update. No full tensors written to disk.
- Lightweight interval counters summarize every update's preclip global norm:
  mean,max,count,fraction above clip threshold. Reuse norm returned by existing
  clipping. Sampled reports capture materialized preupdate parameter snapshots
  and raw gradient RMS, then actual post-AdamW update RMS/weight RMS by tensor,
  with shape/element counts for later layer aggregation. Nearzero-weight ratios
  are null; absolute RMS remains. Initial row has no gradients/updates.
- Each report evaluates deterministic fixed train-corpus/validation windows using
  independent host RNGs and mode restoration. Probes are separate from ordinary
  val CSV. Records include snapshot/reduction/probe time, excluding deferred
  backward/compilation, file I/O and per-update norm/synchronization overhead.
  Extra backward work is not performed; raw gradients come from current train batch.
- Added sequential_gpt_mlx_diagnostics.toml: reference full attention, corrected
  math, T256,d128,L4,H4,batch10, compiled forward/backward, existing512M cache,
  200k total steps,2000warmup, cosine3e-4→3e-5. Eval250, stats1000×2probe batches,
  generation1000 with longer prompts; output gpt-mlx-basic-diagnostics-01.
  Cache/data manifest matches and budget512M targets/202statistics rows checked.
  No fitting/convergence outcome claimed. Existing4K configs untouched.
- Updated MLX stage README with command, fields/cadence/interpretation, and
  exploration backlog to prioritize basic fitting before4K/kernel work.
- Verification:30 focused CPU tests passed (2.30s), covering scalar math vs known
  gradients/weight deltas, interval reset/clipping fraction, deterministic probes,
  disabled mode/validation, JSONL CLI wiring/resume and flush, and exact identical
  weights/losses with stats on/off for eager and compiled tiny runs. Existing
  checkpoint, Torch, progress tests passed; git diff --check passed.
- No substantial training/download/GPU benchmarking launched. User to run new
  config and inspect fixed losses, clipping, layer gradients/updates, and report
  timing. Activation/noise-scale/token-error collectors and viewer remain unimplemented.

## 2026-11-17 — Prepare matched double-LR diagnostics experiment

- Created sequential_gpt_mlx_diagnostics_lr2x.toml from current diagnostics config:
  peak6e-4/floor6e-5, fresh output base gpt-mlx-basic-diagnostics-lr2x-01.
  Model/data/seed/batch/budget/warmup/schedule/reporting all unchanged.
- Parsed both TOMLs and asserted only LR endpoints and output directory differ;
  git diff --check passed. No training launched. User to run fresh (no --resume),
  then compare fixed probe losses, clipping and per-tensor relative updates at
  matched steps/tokens against baseline.

## 2026-11-17 — Analyze baseline and double-LR diagnostics runs

- User requested statistics analysis and notebook graphs. Read both completed
  gpt-mlx-basic-diagnostics(-lr2x)-01 runs:202statistics rows,802loss rows each,
  200k updates/512M tokens,1.85M parameters. Verified matching model, data/cache,
  tokenizer, seeds, budget, sampling, reporting, initial probe losses and tensor
  RMS; only training-config differences are2x peak/floorLR.
- Baseline/double final main val4.490215/4.497227; best4.441736/4.439418 both at
 185500; last10k means4.473583/4.484118. DoubleLR wins early(10k5.46765vs5.57755)
  then trails through middle(100k4.76470vs4.67098), finishes similar. No robust
  final advantage established. Probe val slightly favors double at endpoint but
  fixed train probe favors baseline; smaller probes use different windows.
- Weighted clipping fractions99.4165%/99.3600%; last50k mean preclip norms9.499/
 7.976. Raw gradients grow; blocks all have nonzero sampled gradient/update RMS.
  Final block update RMS~7.2e-6/~1.4e-5. No observed block-wide gradient disappearance;
  activation gradients/directions and gradient noise not collected.
- First attention-projection bias(128elements) contributes late sampled mean
 26.6%/26.0% of squared gradient norm; token embedding29.5%/38.0%. Final LayerNorm
  scale RMS1→14.34/15.91; token embedding final RMS0.0557/0.0994. Measured scale
  drift suggests activation/logit probes, not a confirmed failure mechanism.
- Suggested next controlled experiment: baseline LR schedule with grad_clip5
  instead of1; frequent clipping alone does not prove harm, especially under Adam.
  No inference that accumulation helps without noise/direction probes. No clear
  positive train/val probe gap, but no capacity diagnosis from that observation.
- Added notebooks/06_training_statistics.ipynb with matched-run assertions,
  tables,6embedded figures(loss/deltas/probes,norm/clipping,layer heatmaps,
  gradient-energy shares,scale drift), observations/inferences and proposed next
  test. Uses numpy,matplotlib,stdlib only; reads logs without model loading.
  Aggregated RMS uses element-weighted squared values; gradient shares use squared
  norm contributions. Updated notebook index and MLX README link.
- Installed already-declared analysis dependency matplotlib3.11.2 in .venv from
  cached wheels. Executed all8notebook code cells with stdout/PNG output capture;
  checked notebook JSON/syntax/execution outputs and visually inspected all6plots.
  Standalone PNGs under ignored artifacts/analysis/training-statistics/; notebook
  embeds graphs (~1.79MB). git diff --check passed. No training/evaluation run.
- Statistics JSONL~2.38MB/run; summed timed collector sections8.41/8.32sec versus
  total79.98/78.87min (~0.18%). This excludes sync/norm-transfer/I/O and does not
  measure incremental overhead; no stats-off speed comparison made.

## 2026-11-17 — Clip5 matched experiment and extended training probes

- User authorized grad_clip experiment config/command plus gradient-noise,
  activation and logit logging before next user-run training. Created
  sequential_gpt_mlx_diagnostics_clip5.toml with baseline3e-4→3e-5 schedule,
  T256,L4,d128,batch10,200k updates unchanged; only training-setting change clip1→5.
  Output base gpt-mlx-basic-diagnostics-clip5-01; new diagnostic flags enabled.
- Optional fixed-weight gradient-noise probes:K4 raw batch gradients every5000
  (also step0,1,final), independent host seed+4, same windows across reports.
  Logs mean-gradient norm, average batch norm, unbiased batch variance trace,
  squared-signal estimate ||gbar||²−S/K, noise/signal ratio, token-scaled noise
  estimate, mean pairwise cosine among nonzero gradients and probe timing.
  Nonpositive signal yields null ratios; no forced positive estimate. No clipping,
  Adam transform or optimizer updates. K4 is descriptive and tokens are correlated;
  not an exact optimal accumulation factor. 42reports/168extra backward passes.
  Reference read: https://arxiv.org/html/1812.06162v1 (simple noise-scale estimator).
- Added optional eager activation observer to GPT/Block, preserving identical
  unobserved forward arithmetic. On first fixed train-probe batch, log mean/std/
  RMS/absmax of embedding, normalized branch inputs, branch outputs, residuals,
  final norm. Logits get distribution/vocabulary std, entropy, max/target
  probabilities and top1 accuracy. Scalars only, no persistent tensors.
  Probes preserve model train/eval mode and training batch RNG.
- Extra options default off for older configs. Statistics schema2 retains existing
  fields, adds optional probes.activations/probes.logits/gradient_noise. Updated
  notebook06 loader to accept schemas1/2 without changing saved old-run plots.
  README documents command/cadence/fields/formula/finite-sample limits and overhead.
- Verification:39 focused tests passed (2.49s), including known-vector noise math,
  identical/opposed/zero gradients, uniform-logit entropy/probabilities, observer
  forward equality, mode restoration on failed probe, exact eager/compiled
  training weights/losses with all probes on/off, and CLI logging/resume with
  new flags. Existing checkpoint, Torch, progress, attention, numerical checks
  passed. Config comparison/cache identity/budget, Python/notebook syntax and
  git diff --check passed. No long training or GPU speed/quality claim.
- User to launch fresh clip5 run and compare existing stats against baseline;
  new fields have no historical counterparts. No accumulation implemented yet.

## 2026-11-17 — Analyze completed clip5 run and new probes

- User requested results check. Verified gpt-mlx-basic-diagnostics-clip5-01 completed
  200k updates with202stats/802loss rows; model/data/cache/tokenizer, measured
  initial tensors/probe losses match baseline; only training-config change clip1→5.
- Baseline/clip5 final val4.490215/4.495202; best4.441736/4.451872 at185500; last10k
  means4.473583/4.481474. Clip5 overall clipfraction61.438% vs99.4165%, but last50k
  clip5 also100%. No held-out improvement from loosening clipping; modestly worse
  main validation throughout most trajectory. Final small probe val favorsclip5
  by0.0047 but train probe slightly worse; no robust advantage established.
-42noise reports with four fixed2560token batches, all squared-signal estimates
  positive. Last50k median raw noise/signal5.59994, meanpairwisecos median0.15066;
  final ratio7.77506/cos0.11338/token-noisescale19904. Small-sample raw gradients,
  correlated tokens, one seed and reused probes do not establish exact optimal
  Adam/accumulation batch size. Supports testing accumulation4 (10240tokens/update),
  reverting baselineclip1 and token-matching budget512M/50k updates+LR schedule;
  no implementation or new training undertaken.
- First attention activation branch increasingly cancels embedding input: final
  input/output RMS0.05476/0.05356,sum0.01855; flattened cosine inferred exactly from
  RMS identity−0.94158 vs~0 atinitialization. Not proof of bug/information loss;
  possible normalization-scale sensitivity worth exploring later.
- Global logit std16.519 vs mean within-position vocabstd1.403; entropy4.820nats,
  mean maxprob0.2492,top1accuracy0.2648 on fixed training probe. Raw scale alone
  does not establish probability saturation. No calibration or held-out accuracy
  measured. Final block update RMS~7.0–7.2e-6; blocks remain active.
- Extended notebook06 with clip5 comparison/noise/activation/logit sections and
  four additional figures; prior analysis retained. Executed all12code cells,
  embedded10figures (~2.53MB), checked JSON/syntax/output integrity, visually
  inspected four newfigures, updated notebook index. StandalonePNG artifacts
  refreshed under artifacts/analysis/training-statistics. Whitespace check passed.
  Only log analysis; no model load/training/evaluation executed.
- Clip5 runtime80.53min; timed stats16.50s including noise4.89s; JSONL3.22MB.
  Does not isolate incremental overhead (norm transfer/sync/I/O excluded).

## 2026-11-17 — Implement MLX accumulation and prepare matched factor4 run

- User authorized gradient accumulation config/command; implemented optional
  training.gradient_accumulation_steps (positive integer, default1). Effective
  batch packed once and split into equal microbatches, averaging losses/gradients
  at fixed weights, materializing running sums to bound lazy graphs, clipping
  once, then one AdamW update. Steps/schedule/reporting count optimizer updates.
- Sequential epoch boundaries/tails, preparation budgets, throughput/token counts,
  and metadata now use effective batch B*A. Validation and fixed forward probes
  retain microbatch B to preserve existing windows. Gradient-noise probes average
  microbatch gradients for each effective-batch sample and report B*A*T tokens.
  Metadata adds effective_batch_size and tokens_per_update.
- Resume requires same accumulation factor, defaults missing old checkpoint field
  to1, and restores only completed optimizer updates, so no partial sums saved.
- Added sequential_gpt_mlx_diagnostics_accum4.toml: T256,L4,d128,H4 unchanged;
  microbatch10,A4→40sequences/10240tokens per update;50k updates/512M targets;
  warmup500,decay50000,LR3e-4→3e-5,clip1. Stats/checkpoint/generation250updates,
  noise1250,eval125 (every other baseline eval by tokens). Extended probes enabled.
  Adam betas/weight decay retained per update; fewer total updates intentionally.
  Output gpt-mlx-basic-diagnostics-accum4-01. Existing token cache identity and
  token budget match; no preparation/download needed. Fresh run, no --resume.
- Updated MLX README with command, effective-batch semantics, token-matched
  schedule/reporting, probe behavior, cache/epoch/resume handling. No accumulation
  changes to independent Torch trainer; no optimizer/LR scaling beyond config.
- Verification:44 focused tests passed(3.46s): accumulated gradients vs actual
  large-batch reference(eager/compiled), clip-once-after-average, sequential
  input/target coverage and wrap, tokens/optimizer counts, invalid factors,
  preparation cache budget includes accumulation, exact interrupted/resumed
  random/sequential compiled accumulated runs, rejection of factor changes and
  old checkpoint compatibility. Extended stats on/off still give identical
  training with accumulation; noise token counts effective. Existing attention,
  numerics, progress and Torch checks passed; syntax/config/cache/whitespace
  checks passed. No substantial training launched or quality/speed result claimed.
- User to run accumulation4 then compare losses by processed tokens against
  baselineclip1. Existing notebook step axes must account for4x tokens/update
  before adding the new run to analysis; historical baseline sections unchanged.

## 2026-11-17 — Analyze completed accumulation4 run

- User requested results check. Accum4 completed50k updates/512M targets with
  202stats/402metric rows. Verified same model/data/cache/tokenizer and measured
  initialization as baseline; same LR/clip1 with token-matched warmup/decay.
- Final main validation4.364533 vsbaseline4.490215; best4.350935 at492.8M tokens
  vs4.441736 at474.88M. Final perplexity78.61265 vs89.14059 (~11.81%lower).
  Accum slower early (0–128M matched mean delta+0.13882), then wins all300matched
  evaluations after128M. Matched last25.6M mean delta−0.110204 (20evals); whole
  last128M delta−0.106729. Fixed train/heldout probes4.261803/4.177821 vsbaseline
  4.371099/4.289553. Clear descriptive improvement for this seed/budget, not
  proof of seed-robustness or acceptable sample coherence (samples still weak).
- Overall clipping94.964%, again100%late; last25.6M interval normmean3.624 vs
  baseline10.563. Final LayerNorm weightRMS5.475vs14.336; firstattention activation
  cosine−0.446 vsclip5−0.942. Larger batch follows different scale trajectory;
  smaller raw norms cannot be attributed solely to variance reduction.
- Noise at identical initialization: ratio0.1322 vsclip5(singlemicrobatch)0.5112,
  consistent with4x averaging. Late median ratio5.07185/cos0.1660 vsclip5
  5.59994/0.1507 at different trained weights. Finalratio6.9957/token-scale71635.7
  is not evidence to set batchsize71k. Four fixed samples and raw-gradient estimate
  remain limited; late noise has not disappeared. Baselineclip1 lacks noise logs.
- Important mechanism ambiguity:4x fewer Adam steps at same betas/weightdecay
  changes token-scale momentum and cumulative regularization. Calculated isolated
  matrix decay product(exp sum log1p(-LR*WD))0.0369926baseline vs0.438559accum;
  this is not observed total weight shrinkage (gradients also act; LN vectors
  excluded). Future accumulated WD0.4 run could approximately restore baseline
  decay budget; repeat seed can check reliability. No new config/training started.
- Extended notebook06 with token-aligned4run comparison, matched deltas, noise
  and activation-scale plots. Executed all15code cells, embedded13figures(~3.22MB),
  visually checked3newfigures, checked JSON/syntax/output integrity. Updated index
  to explain accum_name and token comparison. No model loading/evaluation/training.
  Whitespace check passed. Prior notebook results retained.
- Observed runtime84.97min vsbaseline79.98; timed collector34.87sec. Not a controlled
  speed benchmark. Accum4 is current best quality baseline for next experiments;
  no claim that further accumulation/longer training will necessarily improve it.

## 2026-11-17 — Explain gradient-noise estimator and paper interpretation

- User requested calculation/paper takeaway/batch estimate explanation before
  further experiments. Re-read source2.2–2.6 andappendixA.1/A.2:
  https://arxiv.org/html/1812.06162v1 . Reviewed implemented estimator against it.
- Explained frozen-weight raw batch gradients, unbiased batch variance trace
  V=sum||gi-gbar||²/(K−1), corrected squared signal Q=||gbar||²−V/K, ratioV/Q,
  simple noise scale B*V/Q. Ratios are variance/squared-signal, not amplitude;
  negative/nonpositive Q is finite-sample uncertainty, logged null.
- Units: independent sampled windows are underlying examples; token scale is
  window count×fixedcontextlength, not an assumption of token independence.
  At same initial weights: B2560 ratio0.511→scale1309;B10240 ratio0.132→scale1354.
  Noise ratio decreases withbatch while underlying scale is approximatelystable.
- Prior19.9ktoken value was endpointclip5 estimate2560*7.775;would suggest
  ratio~1 at20k only at sameweights/distribution. Not a globally optimal batch.
  Current evolved-model estimate can differ, plus four fixedbatches are noisy.
- Paper predicts critical diminishing-returns batchscale at order-of-magnitude,
  distinguishes simplified traceΣ/||G||² from curvature-aware fullscale. Study
  tunes LR perbatch and does not directly prescribe lowest-validation-loss batch
  at fixedtoken budget. Parallel step savings are not serial accumulation speedup.
- Paper averages signal/variance estimates separately before ratio; our logger
  records lightweight four-batch point estimates (ratio biased/noisy). More probes
  or averaging components can improve reliability; no logging changes made here.

## 2026-11-17 — Prepare accumulation8 comparison config

- User requested factor8 config. Copied accumulation4 to
  sequential_gpt_mlx_diagnostics_accum8.toml: microbatch10,A8,effective80sequences/
  20480tokens;25k updates and250warmup,decay25k; LR3e-4→3e-5,clip1,WD0.1 unchanged.
  Stats/checkpoint/generation125updates,noise625 preserve token spacing;eval125
  corresponds to every other accumulation4 eval by tokens. All probes retained.
  Output gpt-mlx-basic-diagnostics-accum8-01; fresh run using existing cache.
- Parsed/compared TOMLs: only accumulation/step counts/report intervals/output
  differ; model/data/seed/optimizer parameters unchanged. Verified512Mtarget
  budget/cache identity and noise cadence divisibility. No code changes or
  training executed; no additional tests needed for config-only change.

## 2026-11-17 — Save gradient-noise explanation as Markdown

- User explicitly requested prior explanation in Markdown because formulas did
  not render in conversation. Added docs/pretraining/gradient_noise.md using
  note-template structure: frozen-weight noise intuition, V/Q estimator and
  correction, B*V/Q batch scale,19.9k worked example, ratio-vs-scale initialization
  table, paper takeaway, code differences, measured observations/open questions.
  Uses standalone $$ LaTeX blocks plus a plain-text formula fallback.
- Linked note from pretraining docs index and statistics module docstring.
  Checked balanced math/code fences, local links, Python docstring syntax and
  git diff --check. No runtime behavior changes, training or evaluation.

## 2026-11-17 — Commit and push optimization/diagnostics work

- User requested commit+push while their latest experiment trains. Reviewed all
  pending changes: MLX warmup/cosine schedules and LR overrides on resume;
  per-update train display; optional gradient/update/probe statistics, activation/
  logit and gradient-noise collection; gradient accumulation with token-correct
  preparation, progress and checkpoint compatibility; matched experiment configs;
  executed notebook06, gradient-noise note and updated instructions/backlog.
- Final verification for this snapshot:41 focused CPU tests passed(4.67s) across
  accumulation, statistics, pretraining, checkpoint, progress, prepare CLI and
  attention. Notebook15code cells/13embedded figures and new Python/config syntax
  validated; git diff --check passed. Did not rerun Metal/GPU checks or notebooks
  while user training is active; saved graph executions were verified earlier.
- Include source/configs/tests/docs/notebook+outputs and this log; ignored raw
  training artifacts, caches, checkpoints and local kernel assets remain local.
  No active run/config modified. Commit/push requested to origin/main.
- User-run work remaining: finish active experiment, analyze at matched token
  budgets, and choose next LR/weight-decay comparison. Discussed doubling LR and
  halving WD to preserve decay product; no such config/run created yet. Existing
  single-seed comparisons do not isolate accumulation's mechanism.
