"""Token packing and random next-token batches for small in-memory corpora."""

from collections.abc import Iterable

import random

import mlx.core as mx
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
    starts = mx.array([generator.randrange(tokens.size - context_length)
                       for _ in range(batch_size)], dtype=mx.int32)
    windows = starts[:, None] + mx.arange(context_length + 1)
    batch = tokens[windows]
    return batch[:, :-1], batch[:, 1:]
