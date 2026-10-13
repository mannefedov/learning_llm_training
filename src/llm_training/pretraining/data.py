"""Token packing and random next-token batches for small in-memory corpora."""

from collections.abc import Iterable

import torch
from tokenizers import Tokenizer


def encode_documents(texts: Iterable[str], tokenizer: Tokenizer, eos_token: str) -> torch.Tensor:
    """Concatenate documents, appending EOS explicitly after each one.

    Context windows may cross document boundaries. EOS marks boundaries but
    does not reset attention. Both Python IDs and the tensor occupy memory
    while preparing the corpus; this is intended for small samples.
    """
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f"Tokenizer has no EOS token {eos_token!r}")
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special_tokens=False).ids)
        ids.append(eos_id)
    return torch.tensor(ids, dtype=torch.long)


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample CPU token windows; targets are the same windows shifted by one."""
    if batch_size < 1 or context_length < 1:
        raise ValueError("batch_size and context_length must be positive")
    if tokens.ndim != 1 or tokens.numel() <= context_length:
        raise ValueError("Corpus must contain at least context_length + 1 tokens")
    starts = torch.randint(tokens.numel() - context_length, (batch_size,), generator=generator)
    windows = starts[:, None] + torch.arange(context_length + 1)
    batch = tokens[windows]
    return batch[:, :-1].to(device), batch[:, 1:].to(device)
