"""Tiny CPU checks; no remote data or substantial training."""

from dataclasses import asdict
import csv
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from llm_training.models.gpt import GPT, GPTConfig
from llm_training.pretraining.data import encode_documents, get_batch
from llm_training.pretraining.train import TrainConfig, train
from llm_training.pretraining.__main__ import train_from_config
from llm_training.pretraining.tracking import log_run
from llm_training.tokenization.bpe import train_byte_level_bpe


class PretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_causal_attention_cannot_see_future_tokens(self):
        torch.manual_seed(1)
        model = GPT(GPTConfig(vocab_size=16, context_length=8, n_layers=2, n_heads=2, d_model=16))
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 9, 8]])
        logits, loss = model(first, first)
        changed_logits, _ = model(second)
        self.assertEqual(logits.shape, (1, 5, 16))
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(logits[:, :3], changed_logits[:, :3])
        self.assertIs(model.token_embedding.weight, model.lm_head.weight)

    def test_packing_eos_and_shifted_batches(self):
        tokenizer = train_byte_level_bpe(["hello world"], vocab_size=300, show_progress=False)
        texts = ["hello", "world"]
        tokens = encode_documents(texts, tokenizer, "<|endoftext|>")
        eos = tokenizer.token_to_id("<|endoftext|>")
        expected = []
        for text in texts:
            expected.extend(tokenizer.encode(text).ids + [eos])
        self.assertEqual(tokens.tolist(), expected)
        x, y = get_batch(torch.arange(20), 3, 4, torch.device("cpu"))
        self.assertEqual(x.shape, (3, 4))
        torch.testing.assert_close(y, x + 1)
        with self.assertRaises(ValueError):
            get_batch(torch.arange(4), 1, 4, torch.device("cpu"))

    def test_tiny_optimizer_run_reduces_loss(self):
        torch.manual_seed(2)
        model = GPT(GPTConfig(vocab_size=4, context_length=4, n_layers=1, n_heads=1, d_model=16))
        tokens = torch.tensor([0, 1, 2, 3] * 10)
        x, y = get_batch(tokens, 2, 4, torch.device("cpu"))
        initial = model(x, y)[1].item()
        config = TrainConfig(steps=8, batch_size=2, learning_rate=0.02,
                             eval_every=8, eval_batches=1, device="cpu")
        with redirect_stdout(io.StringIO()):
            train(model, tokens, tokens, config)
        self.assertLess(model(x, y)[1].item(), initial)

    @patch("llm_training.tokenization.data.iter_dataset_texts")
    def test_config_training_checkpoint_and_generation(self, iter_texts):
        iter_texts.side_effect = lambda **kwargs: iter(["hello world " * 10])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = train_byte_level_bpe(["hello world"], vocab_size=300, show_progress=False)
            tokenizer.save(str(root / "tokenizer.json"))
            config = root / "train.toml"
            config.write_text(
                '[data]\ndataset_id="local"\nsubset="local"\n'
                'train_split="train"\nval_split="test"\ntrain_documents=1\nval_documents=1\n'
                f'[tokenizer]\npath="{root.as_posix()}/tokenizer.json"\neos_token="<|endoftext|>"\n'
                '[model]\ncontext_length=8\nn_layers=1\nn_heads=2\nd_model=16\n'
                '[training]\nsteps=2\nbatch_size=2\neval_every=2\neval_batches=1\ndevice="cpu"\n'
                f'[output]\ndirectory="{root.as_posix()}/run"\n'
            )
            with redirect_stdout(io.StringIO()):
                checkpoint_path = train_from_config(config)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = GPT(GPTConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model"])
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(asdict(model.config)["vocab_size"], tokenizer.get_vocab_size())
            self.assertTrue((root / "run" / "tokenizer.json").is_file())
            self.assertEqual((root / "run" / "config.toml").read_text(), config.read_text())
            metadata = json.loads((root / "run" / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["device"], "cpu")
            self.assertEqual(metadata["parameter_count"], sum(p.numel() for p in model.parameters()))
            self.assertEqual(metadata["tokenizer_sha256"], hashlib.sha256(
                (root / "run" / "tokenizer.json").read_bytes()
            ).hexdigest())
            with (root / "run" / "metrics.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row["step"]) for row in rows], [0, 1, 2])
            self.assertEqual(rows[0]["train_loss"], "")
            self.assertTrue(all(float(row["val_loss"]) > 0 for row in rows))
            generated = model.generate(torch.tensor([[1] * 8]), max_new_tokens=3)
            self.assertEqual(generated.shape, (1, 11))
            with self.assertRaises(FileExistsError):
                train_from_config(config)

    def test_failed_or_interrupted_run_keeps_flushed_metrics(self):
        for error, status in ((RuntimeError("test failure"), "failed"),
                              (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status), TemporaryDirectory() as temporary:
                output = Path(temporary)
                with self.assertRaises(type(error)):
                    with log_run(output, {"run_name": "test"}) as record:
                        record(0, None, 3.5)
                        # The row is readable before the context manager closes.
                        with (output / "metrics.csv").open(newline="") as file:
                            self.assertEqual(len(list(csv.DictReader(file))), 1)
                        raise error
                metadata = json.loads((output / "metadata.json").read_text())
                self.assertEqual(metadata["status"], status)
                self.assertIn("finished_at", metadata)
                with (output / "metrics.csv").open(newline="") as file:
                    self.assertEqual(list(csv.DictReader(file))[0]["val_loss"], "3.5")


if __name__ == "__main__":
    unittest.main()
