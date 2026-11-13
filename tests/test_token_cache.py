"""Tiny offline checks for budget-limited disk corpus preparation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from llm_training.pretraining.token_cache import cache_identity, open_cache, prepare_cache
from llm_training.tokenization.bpe import train_byte_level_bpe


class TokenCacheTests(unittest.TestCase):
    @patch("llm_training.tokenization.data.iter_dataset_texts")
    def test_budget_stops_stream_reuses_cache_and_checks_identity(self, texts):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = train_byte_level_bpe(["hello world"], vocab_size=300, show_progress=False)
            path = root / "tokenizer.json"
            tokenizer.save(str(path))
            source = dict(dataset_id="local", subset="local", train_split="train", val_split="test",
                          train_documents=20_000_000, val_documents=1, shuffle_buffer_size=0)
            eos = tokenizer.token_to_id("<|endoftext|>")
            first = tokenizer.encode("hello", add_special_tokens=False).ids + [eos]
            second = tokenizer.encode("world", add_special_tokens=False).ids + [eos]
            limit = len(first) + 1
            seen, closed = [], []

            def stream(**kwargs):
                try:
                    for text in ("hello", "world", "never consumed")[:kwargs["num_documents"]]:
                        seen.append((kwargs["split"], text))
                        yield text
                finally:
                    closed.append(kwargs["split"])

            texts.side_effect = stream
            cache = root / "cache"
            prepare_cache(cache, source, path, "<|endoftext|>", limit)
            train, val, manifest = open_cache(cache, cache_identity(source, path, "<|endoftext|>"))
            self.assertIsInstance(train, np.memmap)
            self.assertFalse(train.flags.writeable)
            self.assertEqual(train.tolist(), (first + second)[:limit])
            self.assertTrue(manifest["train"]["last_document_truncated"])
            self.assertEqual(manifest["train"]["documents"], 2)
            self.assertEqual(closed, ["train", "test"])
            self.assertNotIn(("train", "never consumed"), seen)
            texts.reset_mock()
            prepare_cache(cache, source, path, "<|endoftext|>", limit)
            texts.assert_not_called()
            with self.assertRaises(ValueError):
                prepare_cache(cache, source, path, "<|endoftext|>", limit + 1)
            with self.assertRaises(ValueError):
                open_cache(cache, cache_identity({**source, "seed": 9}, path, "<|endoftext|>"))
            with (cache / "train.bin").open("ab") as file:
                file.write(b"bad")
            with self.assertRaises(ValueError):
                open_cache(cache, cache_identity(source, path, "<|endoftext|>"))

    @patch("llm_training.tokenization.data.iter_dataset_texts")
    def test_short_source_never_publishes_complete_cache(self, texts):
        texts.side_effect = lambda **kwargs: (text for text in ["hi"])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = train_byte_level_bpe(["hi"], vocab_size=300, show_progress=False)
            path = root / "tokenizer.json"
            tokenizer.save(str(path))
            source = dict(dataset_id="local", subset="local", train_split="train", val_split="test",
                          train_documents=1, val_documents=1)
            with self.assertRaisesRegex(ValueError, "exhausted"):
                prepare_cache(root / "cache", source, path, "<|endoftext|>", 100)
            self.assertFalse((root / "cache/manifest.json").exists())
            with self.assertRaises(FileExistsError):
                prepare_cache(root / "cache", source, path, "<|endoftext|>", 100)
            prepare_cache(root / "small-cache", source, path, "<|endoftext|>", 100,
                          allow_short_train=True)
            train, _, _ = open_cache(root / "small-cache", cache_identity(source, path, "<|endoftext|>"))
            self.assertLess(train.size, 100)
            texts.reset_mock()
            prepare_cache(root / "small-cache", source, path, "<|endoftext|>", 200,
                          allow_short_train=True)
            texts.assert_not_called()
