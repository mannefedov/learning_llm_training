"""Small offline checks for tokenizer training and saving."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from datasets import IterableDataset
from tokenizers import Tokenizer

from llm_training.tokenization.bpe import train_byte_level_bpe
from llm_training.tokenization.data import iter_dataset_texts
from llm_training.tokenization.__main__ import train_from_config


def tiny_corpus():
    yield {"text": "Hello, world!"}
    yield {"text": " "}
    yield {"text": "Привет, мир!"}
    yield {"text": "This record is outside the three-document sample."}


class TokenizationTests(unittest.TestCase):
    def test_round_trip_unseen_characters_and_serialization(self):
        tokenizer = train_byte_level_bpe(
            iter(["hello world"] * 3), vocab_size=300, show_progress=False
        )
        restored = Tokenizer.from_str(tokenizer.to_str())
        for text in ("hello world", "Привет 🙂 中文", "  spaces\n\ttabs"):
            with self.subTest(text=text):
                self.assertEqual(restored.decode(restored.encode(text).ids), text)
        self.assertIsNotNone(restored.token_to_id("<|endoftext|>"))

    def test_empty_input_and_invalid_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            train_byte_level_bpe([], show_progress=False)
        with self.assertRaisesRegex(ValueError, "alphabet"):
            train_byte_level_bpe(["hello"], vocab_size=256)

    @patch("llm_training.tokenization.data.load_dataset")
    def test_document_limit_and_blank_filtering(self, load):
        load.return_value = IterableDataset.from_generator(tiny_corpus)
        texts = list(iter_dataset_texts(
            dataset_id="local", subset="example", num_documents=3,
            shuffle_buffer_size=0, revision="fixed-revision",
        ))
        self.assertEqual(texts, ["Hello, world!", "Привет, мир!"])
        load.assert_called_once_with(
            "local", name="example", split="train", streaming=True,
            revision="fixed-revision",
        )

    @patch("llm_training.tokenization.data.load_dataset")
    def test_config_run_saves_reloadable_artifact_and_rejects_overwrite(self, load):
        load.return_value = IterableDataset.from_generator(tiny_corpus)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            config = Path(temporary) / "train.toml"
            config_text = (
                '[data]\ndataset_id = "local"\nsubset = "example"\n'
                'num_documents = 3\nshuffle_buffer_size = 0\n'
                '[tokenizer]\nvocab_size = 300\nshow_progress = false\n'
                f'[output]\ndirectory = "{output.as_posix()}"\n'
            )
            config.write_text(config_text, encoding="utf-8")
            artifact = train_from_config(config)
            restored = Tokenizer.from_file(str(artifact))
            text = "Unseen 🙂\n"
            self.assertEqual(restored.decode(restored.encode(text).ids), text)
            self.assertEqual((output / "config.toml").read_text(), config_text)
            load.reset_mock()
            with self.assertRaises(FileExistsError):
                train_from_config(config)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
