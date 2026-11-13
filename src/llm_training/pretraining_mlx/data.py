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


def sequential_batch(tokens, batch_size: int, context_length: int, step: int):
    """Visit each target token once, in order, with one-token overlap for shifting.

    Full batches only; the tail smaller than batch_size * context_length is
    unused. No wrapping or random resampling occurs.
    """
    size = batch_size * context_length
    start = step * size
    if step < 0 or batch_size < 1 or context_length < 1 or start + size >= tokens.size:
        raise ValueError("Sequential batch is outside the corpus")
    window = tokens[start:start + size + 1]
    x = window[:-1].reshape(batch_size, context_length)
    y = window[1:].reshape(batch_size, context_length)
    return mx.array(x), mx.array(y)
