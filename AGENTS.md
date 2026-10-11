# Project guidance

## Purpose

This is a personal educational project exploring all parts of LLM training:
tokenization, architectures, pretraining, fine-tuning, post-training, RL, and
related topics. The goal is understanding and interview preparation.

## Implementation style

- Keep code clean, minimal, readable, and easy to navigate.
- Prefer small explicit implementations over frameworks and abstractions until
  there is a concrete need for them.
- Use clear names and explain non-obvious mathematical or algorithmic choices.
- Make changes in focused increments that the user can inspect and understand.
- Add dependencies only when needed for the current learning task.
- Do not implement future training components merely to fill out the scaffold.

## Organization and references

- Reusable Python code belongs in `src/llm_training/`, grouped by training stage.
- Freeform explorations belong in the top-level `notebooks/` directory.
- `docs/` documents knowledge and work actually done, not suggestions, plans,
  or exercises. Create or expand knowledge docs only on direct user request.
- Do not create speculative notes just because a topic was discussed or researched.
- Put requested examples and exercises in notebooks, with supporting references
  in their Markdown cells until documentation is explicitly requested.
- When docs are requested, organize them to match the code and use
  `docs/note_template.md` as a starting point. Link modules to existing relevant
  notes in docstrings; do not create documentation just to supply a link.
- Record source URLs, the ideas used, implementation differences, and open
  questions. Distinguish reported results from locally observed results.
- Defer experiment-tracking infrastructure until an actual experiment needs it.

## Preserve hands-on learning

- Basic verification is welcome: imports, syntax checks, and small focused
  tests or tiny smoke runs appropriate to the change.
- Leave substantial training, evaluation, sweeps, large downloads, and extended
  notebook execution for the user unless they explicitly ask you to run them.
- Provide exact commands or notebook entry points for work left to the user,
  with a brief explanation of what to inspect or learn.
- Be explicit about what was verified and what remains unrun. Never invent
  experiment outcomes.

## Session log and pushing

- Append a concise entry to `PROGRESS.md` after meaningful work in each session.
- Before pushing work, ensure the progress log includes the changes being
  pushed, verification performed, work left for the user, and next steps.
- Keep previous entries intact; avoid recording credentials or private data.
- Follow the user's current instructions if they change these preferences.
