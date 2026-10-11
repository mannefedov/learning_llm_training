"""Train byte-level BPE using Hugging Face Tokenizers.

An exploratory example lives in notebooks/01_bpe_fineweb2.ipynb.
"""

from collections.abc import Iterable, Sequence
from itertools import chain

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_byte_level_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int = 8_000,
    min_frequency: int = 2,
    special_tokens: Sequence[str] = ("<|endoftext|>",),
    show_progress: bool = True,
) -> Tokenizer:
    """Learn BPE merges from text, preserving case, Unicode, and whitespace.

    The full byte alphabet covers unseen characters. Special tokens are reserved
    in the vocabulary but are not automatically inserted into encoded text.
    Input can be streamed, but training still holds corpus statistics in memory.
    """
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must be unique")
    if vocab_size < len(alphabet) + len(special_tokens):
        raise ValueError("vocab_size must fit the byte alphabet and special tokens")
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")

    iterator = iter(texts)
    first = next(iterator, None)
    if first is None:
        raise ValueError("Cannot train on an empty text iterator")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=alphabet,
        special_tokens=list(special_tokens),
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(chain([first], iterator), trainer=trainer)
    return tokenizer
