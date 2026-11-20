# Findings from the dense GPT experiments

## Comparison points

The dense MLX models use four layers, a 256-token context, and the same 8k
Russian FineWeb2 tokenizer. Loss is held-out cross-entropy. These are single
runs. Compare the same validation windows, training token budget, and schedule
before attributing a difference to a model change.

| Standard model experiment | Training targets | Final validation loss |
| --- | ---: | ---: |
| Width 128, baseline | 512M | 4.4902 |
| Double the baseline LR | 512M | 4.4972 |
| Raise clip threshold from 1 to 5 | 512M | 4.4952 |
| Accumulate 4 microbatches | 512M | 4.3645 |
| Accumulate 8 microbatches | 512M | 4.4162 |
| Accumulate 8, double LR and halve weight decay | 512M | 4.3144 |
| Width 256, native softmax | 512M | 3.8187 |
| Width 256, native softmax, extended schedule | 2.048B | 3.6741 |

The 2B reference used 100k updates, 20,480 targets per update, AdamW with
betas (0.9, 0.95), clip 1, weight decay 0.05, 250 warmup updates, and cosine LR
from 6e-4 to 6e-5. Its best validation loss was 3.6522. The 512M width comparison
also switched softmax implementation. The 2B comparison also stretched cosine
decay. Generated samples from the 512M width-256 run still had grammar and
coherence problems despite the lower loss.

## Optimization and correctness

- **Retune LR when the effective batch changes.** At fixed 512M targets,
  accumulation 8 had less measured raw gradient noise but worse loss than
  accumulation 4 at the original LR. Doubling LR and halving decay at
  accumulation 8 improved loss while preserving LR × decay per update. Changing
  accumulation also changes the number of optimizer updates per token.
  [Gradient noise estimator](../pretraining/gradient_noise.md)
- **Use clipping and scale measurements to diagnose.** Raising the clipping
  threshold lowered the fraction of clipped updates from 99% to 61%, with no
  held-out improvement. Sampled blocks retained nonzero gradient and update RMS.
  Watch gradients before clipping, actual optimizer updates, per-position logit
  spread, and fixed train and validation probes when investigating changes.
- **Test memorization with prompts covered by the training windows.** The
  289-token passage initially left just 34 possible starts at context 256. With
  the passage repeated, the model reached loss 0.00215 in 1,000 updates,
  reproduced all three tested 64-token continuations, and preserved logits on
  checkpoint reload. This checks basic learning and generation on that passage.
- **Treat a short μTransfer sweep as a way to choose candidates.** Our corrected
  zero-decay sweep across widths 32, 64, 128 and 256 put the best tested LRs in
  the 2⁻⁸ to 2⁻⁷ region. It used one seed and 40.96M targets per candidate.
  Test transfer again when changing parameter shapes or routing.
  [Sweep plots](../../notebooks/09_mup_sweep_results.ipynb)

## Attention and performance

- **Short-context sliding windows preserved similar loss in this test.**
  With a window of 64 at context 256, full, 1:1 local/full, and 3:1 local/full
  ended at 4.731, 4.730, and 4.712 after 200k updates. Those small single-run
  differences give little evidence for a particular mix. The reference SWA
  computes tiled bands with O(TW) score work and includes self in the window.
  Verify forward and backward against a dense band mask before timing.
  [Gemma 2](https://arxiv.org/abs/2408.00118) · [Gemma 3](https://arxiv.org/abs/2503.19786)
- **Check whether numerical changes matter in training.** Native and corrected
  softmax/CE in the 3:1 model both finished near 4.691 after 400k updates.
  The corrected path cost more time in the short benchmark below. Fused full
  attention required a supported head shape, so the earlier fused quality run
  also changed the head layout. Hold that layout fixed when measuring a kernel.
- **Separate context, capacity, and kernel changes.** The larger 4K SWA run
  reached 3.848 at 512M targets and 3.679 after another 51.2M at a lower LR.
  This continued training from the same checkpoint and changed the LR. Its
  loss also used different validation windows from the 256-context runs.
- **Measure full training speed as well as kernels.** Forward/backward dominated
  the local profile. At batch 10, context 256, and vocab 8k, native model
  forward/backward took 11.18 ms, corrected softmax/CE 25.20 ms, and the compiled
  corrected version 18.76 ms in a short synchronized benchmark. The patched
  4K window backward passed strict attention checks and showed small gains in
  short paired benchmarks, with no noticeable gain in actual training. Native
  repeat runs also diverged in exact parameters after multiple updates.
  [FlashAttention-2](https://arxiv.org/abs/2307.08691) · [Softmax correction](https://jonbarron.info/softmaxfix/)

## Applying these findings to the next models

Use the dense run as a comparison point. First validate forward/backward and
checkpoint generation on a tiny problem, then tune candidate architectures at
matched short budgets before scaling them up. For MoE, record total and active
parameters, expert utilization, and actual training throughput alongside loss.
For MLA, separate training loss and throughput from decode speed and KV cache
size, and compare attention outputs against a simple reference. Keep data,
tokenizer, context, and validation fixed within each comparison.

## Provenance

The detailed history is in [PROGRESS.md](../../PROGRESS.md). Retired configs,
notebooks and kernel experiments are in Git at `a22d1a8`. On the experiment
machine, the local archive under `artifacts/archive/pre-mup` keeps the raw runs,
checkpoints, plots, and retired source files. The standard 2B comparison is in
`pretraining/gpt-mlx-basic-diagnostics-width256-2b-01` under that archive.
