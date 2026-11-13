"""Stream a bounded text sample from a Hugging Face dataset.

An exploratory example lives in notebooks/01_bpe_fineweb2.ipynb.
"""

from collections.abc import Iterator
import gc

from datasets import load_dataset


def iter_dataset_texts(
    *,
    dataset_id: str,
    subset: str,
    num_documents: int,
    split: str = "train",
    text_column: str = "text",
    seed: int = 42,
    shuffle_buffer_size: int = 100,
    revision: str | None = None,
) -> Iterator[str]:
    """Yield nonblank texts from at most num_documents sampled records.

    Set shuffle_buffer_size to zero to keep dataset order. A finite buffer is
    an approximate shuffle, not a uniform corpus sample. Use a dataset commit
    SHA as revision to hold the data version fixed across runs.
    """
    if num_documents < 1:
        raise ValueError("num_documents must be positive")
    if shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size must be nonnegative")

    dataset = load_dataset(
        dataset_id, name=subset, split=split, streaming=True, revision=revision
    )
    if shuffle_buffer_size:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    records = iter(dataset.take(num_documents))
    try:
        for document in records:
            text = document[text_column]
            if not isinstance(text, str):
                raise TypeError(f"Column {text_column!r} must contain strings")
            if text.strip():
                yield text
    finally:
        # A bounded sample leaves the underlying Parquet stream partly read.
        # Release it while Python is fully alive, rather than at interpreter exit:
        # https://github.com/huggingface/datasets/issues/7467
        close = getattr(records, "close", None)
        if close is not None:
            close()
        del records, dataset
        gc.collect()
