# Gradient noise and batch size

## Learning question

How much do gradients disagree across training batches, how do we measure that
disagreement, and how can it guide the effective batch size?

## References

- [An Empirical Model of Large-Batch Training](https://arxiv.org/html/1812.06162v1)
  — McCandlish et al., 2018. Sections 2–3 explain the noise scale and critical
  batch size; Appendix A.1 describes estimation, and A.2 discusses learning-rate
  tuning across batch sizes.

## Key ideas

### What “noise” means

Keep the model's weights fixed and compute gradients on different random batches.
The gradients will differ because each batch contains different examples. This is
sampling noise, not floating-point error or randomness added by the optimizer.

Think of each batch as making a recommendation about how to change the weights.
Averaging batches combines their shared recommendation and reduces disagreement.
Even with deterministic computation, changing the batch changes the gradient.

### How our collector calculates it

At one fixed set of weights, compute raw gradients on $K$ batches:

$$
g_1, g_2, \ldots, g_K.
$$

Each $g_i$ is conceptually one vector containing every trainable parameter's
gradient. We keep the parameter tensors separate in code, but sum their squared
norms. These are gradients of mean token loss, before clipping or Adam.

First calculate their average:

$$
\bar{g} = \frac{1}{K} \sum_{i=1}^{K} g_i.
$$

Next estimate the batch-gradient variance, summed across all parameter components:

$$
V = \frac{1}{K-1} \sum_{i=1}^{K} \left\|g_i - \bar{g}\right\|_2^2.
$$

If every batch produces the same gradient, $V=0$. Large disagreement produces a
larger $V$. The divisor $K-1$ is the usual sample-variance correction.

The mean gradient still contains sampling noise. Its squared norm therefore
overestimates the squared magnitude of the underlying expected gradient. Correct
for that remaining noise:

$$
Q = \left\|\bar{g}\right\|_2^2 - \frac{V}{K}.
$$

Here $Q$ estimates the squared signal magnitude. The logged ratio is:

$$
r = \frac{V}{Q}.
$$

This is a ratio of variance to **squared** signal, not a ratio of magnitudes.
For example, $r=5$ means noise energy is about five times squared signal
magnitude. The corresponding RMS-noise-to-signal-magnitude ratio is $\sqrt{5}$.

With few batches, $Q$ can be nonpositive. The true squared signal cannot be
negative; the estimate is uncertain. Our collector records the signed estimate
but writes `null` for ratios when $Q \leq 0$.

The variance and squared-signal estimates are unbiased under independent batch
sampling. Their **ratio is not an unbiased estimate**, and can fluctuate strongly
when the signal denominator is small.

### From noise ratio to batch-size scale

At fixed weights and for independently sampled examples, averaging more examples
reduces gradient variance approximately in proportion to the inverse batch size:

$$
V(B) \propto \frac{1}{B}.
$$

The expected gradient stays the same. If the current batch has size $B$ and ratio
$r$, a candidate batch size $B'$ would therefore have approximately:

$$
r(B') \approx r(B)\frac{B}{B'}.
$$

Setting this ratio to one gives the batch scale where noise variance and squared
signal are comparable:

$$
\boxed{B_{\mathrm{simple}} \approx B\,r = B\frac{V}{Q}.}
$$

In our language-model probes, independently sampled **context windows** are the
underlying examples. We convert the window-count scale to tokens using the fixed
context length. This does not assume that tokens within a window are independent.

For the earlier clip-5 run's final measurement:

$$
B = 2{,}560\ \text{tokens},\qquad r \approx 7.775,
$$

$$
B_{\mathrm{simple}} \approx 2{,}560 \times 7.775
\approx 19{,}900\ \text{tokens}.
$$

Holding those weights and the sampling distribution fixed, this predicts:

| Batch tokens | Predicted noise-to-signal ratio |
| ---: | ---: |
| 2,560 | 7.78 |
| 10,240 | 1.94 |
| 20,480 | 0.97 |

This is the reasoning behind testing accumulation ×8. It is a candidate experiment,
not a prescription to use exactly 19,900 tokens throughout training.

### Noise ratio versus noise scale

The ratio changes with batch size. The inferred underlying scale should be
approximately stable when measured at the same weights and data distribution.
Our measurements at identical initialization illustrate the distinction:

| Batch tokens | Measured noise/signal | Estimated token scale |
| ---: | ---: | ---: |
| 2,560 | 0.511 | 1,309 |
| 10,240 | 0.132 | 1,354 |

Quadrupling the batch reduced the measured ratio approximately fourfold, while
the estimated scale remained similar. The estimates differ because we use only
four batches. After training, runs reach different weights, so their signal,
variance, and noise scale can differ substantially.

### What the paper concludes

The paper finds that gradient noise scale predicts the approximate **critical
batch size**: the region where increasing batch size starts giving substantially
diminishing returns in optimizer-step savings. The prediction is at the
order-of-magnitude level.

| Batch relative to the critical scale | Expected tradeoff |
| --- | --- |
| Much smaller | Larger batches can substantially increase progress per update. |
| Around the scale | Diminishing returns become important. |
| Much larger | Additional examples buy relatively little extra progress per update. |

This is a guide to the compute/step tradeoff, not an exact batch size that minimizes
validation loss at a fixed token budget. The paper tunes learning rates across
batch sizes, and distinguishes the inexpensive simple scale from a fuller
curvature-aware quantity. It also finds that the useful scale can grow during
training.

Our serial accumulation does not provide the same hardware speedup as running
larger batches in parallel across many devices. It still performs the forward
and backward computation for every microbatch.

### Formulas without math rendering

```text
K = number of probe batches, all evaluated at identical weights
g_i = raw gradient vector for probe batch i

mean_gradient = sum(g_i) / K
V = sum(squared_l2_norm(g_i - mean_gradient)) / (K - 1)
Q = squared_l2_norm(mean_gradient) - V / K

noise_to_signal = V / Q                         # only when Q > 0
noise_scale_tokens = tokens_per_batch * V / Q    # only when Q > 0

predicted_ratio_at_new_batch = noise_to_signal * current_batch / new_batch
```

## Implementation connection

- [Collector and estimator](../../src/llm_training/pretraining_mlx/statistics.py):
  `TrainingStatistics.gradient_noise()` and `gradient_noise_summary()`.
- [Accumulation](../../src/llm_training/pretraining_mlx/train.py):
  `accumulate_gradients()` averages microbatch gradients before clipping.
- [Run instructions](../../src/llm_training/pretraining_mlx/README.md#clip-threshold-experiment-and-extended-probes).

Our collector uses four separately seeded batches, at fixed current weights,
without optimizer updates. When accumulation is enabled, each probe gradient
represents a full effective batch. The same probe windows are reused across
reports, independently of training's batch RNG.

The paper obtains related estimates from gradient norms before and after
averaging local batches. It stabilizes the signal and variance components with
separate moving averages before taking their ratio. Our logger instead records
four-batch point estimates. More probe batches or averaging components separately
could improve reliability.

We measure raw gradients, not Adam-preconditioned or clipped gradients. The
estimate therefore does not fully describe our optimizer's actual update noise.

## Observations and open questions

The numeric examples above are **local measurements**, not results reported by
the paper. See the
[dense GPT experiment findings](../models/experiment_findings.md) for the saved comparisons.

Accumulation ×4 improved validation at the same 512M-token budget. It also changed
the number of Adam updates, token-scale momentum history, and cumulative weight
decay. Accumulation ×8 lowered sampled gradient noise further but increased
validation loss relative to ×4 at the original LR. Doubling LR and halving
weight decay improved the ×8 result. This supports retuning LR with batch size.
