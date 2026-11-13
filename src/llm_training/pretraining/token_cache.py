"""Stream token IDs to disk and reopen them without loading the corpus into RAM."""

import hashlib
from importlib.metadata import version
import json
import time
from pathlib import Path

import numpy as np


def cache_identity(data_config, tokenizer_path: Path, eos_token: str):
    return {
        "format_version": 1,
        "dtype": "<u4",
        "data_config": data_config,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "tokenizers_version": version("tokenizers"),
        "eos_token": eos_token,
    }


def open_cache(directory: Path, identity: dict):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["identity"] != identity:
        raise ValueError("Token cache settings/tokenizer differ; choose a new cache directory")
    corpora = {}
    for split in ("train", "val"):
        path = directory / f"{split}.bin"
        count = manifest[split]["tokens"]
        if count < 1 or path.stat().st_size != count * 4:
            raise ValueError(f"Invalid or incomplete token cache: {path}")
        corpora[split] = np.memmap(path, dtype="<u4", mode="r", shape=(count,))
    return corpora["train"], corpora["val"], manifest


def prepare_cache(directory: Path, data_config: dict, tokenizer_path: Path,
                  eos_token: str, train_token_limit: int):
    """Write documents until the fixed training token budget is filled.

    The last training document may be truncated; validation uses its document
    limit. Publish a manifest only after both splits finish.

    Interrupted caches are never silently reused or overwritten. Preparation
    must be restarted in a new directory; resume is not implemented.
    """
    from tokenizers import Tokenizer
    from llm_training.tokenization.data import iter_dataset_texts

    if train_token_limit < 2:
        raise ValueError("Training token budget must include at least one input/target pair")
    identity = cache_identity(data_config, tokenizer_path, eos_token)
    if (directory / "manifest.json").exists():
        train_tokens, _, _ = open_cache(directory, identity)
        if train_tokens.size < train_token_limit:
            raise ValueError("Existing cache is too small for this budget; choose a new cache directory")
        print(f"Reusing token cache: {directory}")
        return
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f"Tokenizer has no EOS token {eos_token!r}")
    source = dict(data_config)
    train_split = source.pop("train_split")
    val_split = source.pop("val_split")
    train_documents = source.pop("train_documents")
    val_documents = source.pop("val_documents")
    if train_split == val_split:
        raise ValueError("Use separate train and validation splits")
    if directory.exists():
        raise FileExistsError(f"Incomplete cache exists; choose a new cache directory: {directory}")
    directory.mkdir(parents=True)
    manifest = {"identity": identity}
    for name, split, limit in (("train", train_split, train_documents), ("val", val_split, val_documents)):
        count = documents = 0
        truncated = False
        token_limit = train_token_limit if name == "train" else None
        last_report = time.monotonic()
        print(f"Preparing {name}: up to {limit:,} document records"
              + (f", stopping at {token_limit:,} tokens" if token_limit else ""), flush=True)
        texts = iter_dataset_texts(**source, split=split, num_documents=limit)
        try:
            with (directory / f"{name}.bin").open("xb") as file:
                for text in texts:
                    ids = tokenizer.encode(text, add_special_tokens=False).ids
                    ids.append(eos_id)
                    if token_limit is not None:
                        remaining = token_limit - count
                        truncated = len(ids) > remaining
                        ids = ids[:remaining]
                    np.asarray(ids, dtype="<u4").tofile(file)
                    count += len(ids)
                    documents += 1
                    if time.monotonic() - last_report >= 5:
                        print(f"{name}: {documents:,} documents; {count:,} tokens; "
                              f"{count * 4 / 2**30:.2f} GiB", flush=True)
                        last_report = time.monotonic()
                    if token_limit is not None and count == token_limit:
                        break
        finally:
            texts.close()
        if not count:
            raise ValueError(f"No tokens in {name} data")
        if token_limit is not None and count < token_limit:
            raise ValueError(f"Document limit/source exhausted: got {count:,} of {token_limit:,} tokens; "
                             "increase train_documents or reduce the training budget and use a new cache directory")
        manifest[name] = {"documents": documents, "tokens": count,
                          "last_document_truncated": truncated}
        print(f"{name}: saved {documents:,} documents / {count:,} tokens", flush=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
