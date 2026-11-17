# Explore later

2026-11-17: focus current experiments on fitting a basic 256-token full-attention
model and collecting training diagnostics. Keep 4K context and attention-kernel
optimization parked until the basic training behavior is understood.

## Fused attention backward on MLX

Parked 2026-11-16: user tried the NAX backward in actual training and observed
no noticeable speedup. Keep the existing tiled backward as the default.

We implemented native causal-window backward, then on-chip FP32/D64 backward
using SIMD-group matrices and M5 TensorOps. Short paired benchmarks suggested
small gains with NAX16, but they did not establish a practical training gain.
Strict attention and initial model-gradient checks passed; multi-update parameter
identity failed even for repeated native controls. Long-run quality is untested.

Resume from [the experiment instructions](../../experiments/mlx_swa/README.md):
patch order, switches, tests and source revision are recorded there. Next compare
matched real training runs (including reporting/evaluation overhead), then use
GPU counters to investigate register spills, occupancy, reuse and pipelining.
Raw results and intermediate reports are local-only under
`.local/mlx-swa-reports-2026-11-16/`.

## Long-document continued pretraining

Parked 2026-11-16: initialize from the current 4K model after its run finishes,
then train on 4K windows drawn entirely from individual long documents. Keep
model dimensions and tokenizer unchanged. Each row stays within one document,
so this experiment needs no cross-document attention mask.

Existing training cache (complete EOS-delimited documents, excluding EOS):

| Minimum length | Documents | Tokens |
| --- | ---: | ---: |
| 4,096 | 18,990 | 194,915,631 |
| 8,192 | 7,620 | 130,997,345 |

The ≥4K subset provides 39,972 complete 4K windows: about 20k steps at batch 2.
This reuses previously seen training data with more emphasis on long documents.

When resumed:

- Add document-aware window indexing; keep shifted targets within the document
  (EOS can be the final target), rather than concatenating selected documents
  and letting ordinary sequential batches cross their boundaries again.
- Add explicit initialization from a checkpoint with a fresh optimizer/step
  counter and configurable lower learning rate. Keep strict `--resume` for exact
  continuation; it currently rejects a different corpus.
- Expand held-out long-document validation: the current validation cache has
  only four documents ≥4K and none ≥8K. Keep training documents out of it.
- Compare validation loss and long-prompt continuations before and after this
  stage. No corpus download or continuation run has been started.

## BF16 mixed-precision training

Parked 2026-11-16: try BF16 computation with FP32 master weights and optimizer
state. Keep suitable precision for normalization, reductions, and loss.

Motivation: local step profiling found that
forward/backward takes roughly 94–96% of step time. At batch 2, isolated BF16
components were faster: full attention 16.1→8.4 ms, local attention 4.4→3.0 ms,
MLP 3.8→2.3 ms, and output projection/loss 7.9→5.1 ms.

Next: add an opt-in implementation, check gradients and optimizer/checkpoint
state, then compare matched short training runs for throughput, memory, loss,
and stability. Component timings do not establish whole-training speed or quality.
No production precision change has been made.
