"""Next-token batches from in-memory arrays or disk-backed token corpora."""

from collections.abc import Iterable

import random

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer


def encode_documents(texts: Iterable[str], tokenizer: Tokenizer, eos_token: str) -> mx.array:
    """Concatenate documents, appending EOS explicitly after each one.

    Context windows may cross document boundaries. EOS marks boundaries but
    does not reset attention. Both Python IDs and the array occupy memory
    while preparing the corpus; this is intended for small samples.
    """
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f"Tokenizer has no EOS token {eos_token!r}")
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special_tokens=False).ids)
        ids.append(eos_id)
    return mx.array(ids, dtype=mx.int32)


def get_batch(tokens: mx.array, batch_size: int, context_length: int,
              generator: random.Random | None = None) -> tuple[mx.array, mx.array]:
    """Sample shifted windows with an independent host RNG (no model RNG changes)."""
    if batch_size < 1 or context_length < 1:
        raise ValueError("batch_size and context_length must be positive")
    if tokens.ndim != 1 or tokens.size <= context_length:
        raise ValueError("Corpus must contain at least context_length + 1 tokens")
    generator = generator if generator is not None else random.Random()
    starts = [generator.randrange(tokens.size - context_length) for _ in range(batch_size)]
    if isinstance(tokens, np.ndarray):
        # Index only the requested windows on the host; never convert the memmap.
        windows = np.array(starts)[:, None] + np.arange(context_length + 1)
    else:
        windows = mx.array(starts, dtype=mx.int32)[:, None] + mx.arange(context_length + 1)
    batch = tokens[windows]
    return mx.array(batch[:, :-1]), mx.array(batch[:, 1:])


def batches_per_epoch(tokens, batch_size: int, context_length: int) -> int:
    if batch_size < 1 or context_length < 1 or tokens.ndim != 1:
        raise ValueError("Expected a one-dimensional corpus and positive batch/context sizes")
    count = (tokens.size - 1) // (batch_size * context_length)
    if count < 1:
        raise ValueError("Sequential corpus needs at least batch_size * context_length + 1 tokens")
    return count


def sequential_batch(tokens, batch_size: int, context_length: int, step: int):
    """Read the global zero-based step, wrapping after each full sequential pass.

    Windows overlap by one token for shifted targets. A tail smaller than a
    full batch is unused in each epoch; no padding or random resampling occurs.
    """
    if step < 0:
        raise ValueError("Sequential step must be nonnegative")
    epoch_steps = batches_per_epoch(tokens, batch_size, context_length)
    size = batch_size * context_length
    start = (step % epoch_steps) * size
    window = tokens[start:start + size + 1]
    x = window[:-1].reshape(batch_size, context_length)
    y = window[1:].reshape(batch_size, context_length)
    return mx.array(x), mx.array(y)
