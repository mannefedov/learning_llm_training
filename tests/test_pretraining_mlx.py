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

import importlib.util
import random

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("Install the pretraining-mlx extra on Apple silicon")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from llm_training.models.gpt_mlx import GPT, GPTConfig
from llm_training.pretraining_mlx.data import encode_documents, get_batch
from llm_training.pretraining_mlx.train import TrainConfig, train
from llm_training.pretraining_mlx.__main__ import train_from_config
from llm_training.pretraining_mlx.tracking import log_run
from llm_training.tokenization.bpe import train_byte_level_bpe


class PretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.previous_device)

    def test_causal_attention_cannot_see_future_tokens(self):
        mx.random.seed(1)
        model = GPT(GPTConfig(vocab_size=16, context_length=8, n_layers=2, n_heads=2, d_model=16))
        first = mx.array([[1, 2, 3, 4, 5]])
        second = mx.array([[1, 2, 3, 9, 8]])
        logits, loss = model(first, first)
        changed_logits, _ = model(second)
        self.assertEqual(logits.shape, (1, 5, 16))
        self.assertTrue(mx.isfinite(loss).item())
        self.assertTrue(mx.allclose(logits[:, :3], changed_logits[:, :3], atol=1e-6).item())
        self.assertFalse(any("lm_head" in name for name, _ in tree_flatten(model.parameters())))

    def test_packing_eos_and_shifted_batches(self):
        tokenizer = train_byte_level_bpe(["hello world"], vocab_size=300, show_progress=False)
        texts = ["hello", "world"]
        tokens = encode_documents(texts, tokenizer, "<|endoftext|>")
        eos = tokenizer.token_to_id("<|endoftext|>")
        expected = []
        for text in texts:
            expected.extend(tokenizer.encode(text).ids + [eos])
        self.assertEqual(tokens.tolist(), expected)
        x, y = get_batch(mx.arange(20), 3, 4, random.Random(42))
        self.assertEqual(x.shape, (3, 4))
        self.assertTrue(mx.array_equal(y, x + 1).item())
        with self.assertRaises(ValueError):
            get_batch(mx.arange(4), 1, 4, random.Random(42))

    def test_tiny_optimizer_run_reduces_loss(self):
        mx.random.seed(2)
        model = GPT(GPTConfig(vocab_size=4, context_length=4, n_layers=1, n_heads=1, d_model=16))
        tokens = mx.array([0, 1, 2, 3] * 10)
        x, y = get_batch(tokens, 2, 4, random.Random(42))
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
                '[generation]\nenabled=true\nevery=1\nprompts=["hello"]\nmax_new_tokens=3\n'
                f'[output]\ndirectory="{root.as_posix()}/run"\n'
            )
            with redirect_stdout(io.StringIO()):
                checkpoint_path = train_from_config(config)
            checkpoint = json.loads(checkpoint_path.read_text())
            model = GPT(GPTConfig(**checkpoint["model_config"]))
            model.load_weights(str(checkpoint_path.parent / checkpoint["model"]))
            from llm_training.pretraining_mlx.generate import load_checkpoint, generate_text
            loaded, saved_tokenizer, eos_id = load_checkpoint(checkpoint_path.parent, device="cpu")
            probe = mx.array([[1, 2, 3]])
            self.assertTrue(mx.array_equal(model(probe)[0], loaded(probe)[0]).item())
            sample = generate_text(loaded, saved_tokenizer, "hello", max_new_tokens=3,
                                   temperature=0.8, seed=7, eos_id=eos_id)
            self.assertEqual(sample, generate_text(loaded, saved_tokenizer, "hello",
                             max_new_tokens=3, temperature=0.8, seed=7, eos_id=eos_id))
            self.assertTrue(sample.startswith("hello"))
            state = tree_unflatten(mx.load(str(checkpoint_path.parent / checkpoint["optimizer"])))
            self.assertEqual(state["step"].item(), 2)
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(asdict(model.config)["vocab_size"], tokenizer.get_vocab_size())
            self.assertTrue((root / "run" / "tokenizer.json").is_file())
            self.assertEqual((root / "run" / "config.toml").read_text(), config.read_text())
            metadata = json.loads((root / "run" / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["device"], "cpu")
            self.assertEqual(metadata["parameter_count"], sum(p.size for _, p in tree_flatten(model.parameters())))
            self.assertEqual(metadata["tokenizer_sha256"], hashlib.sha256(
                (root / "run" / "tokenizer.json").read_bytes()
            ).hexdigest())
            with (root / "run" / "metrics.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row["step"]) for row in rows], [0, 1, 2])
            self.assertEqual(rows[0]["train_loss"], "")
            self.assertTrue(all(float(row["val_loss"]) > 0 for row in rows))
            generated = model.generate(mx.array([[1] * 8]), max_new_tokens=3)
            self.assertEqual(generated.shape, (1, 11))
            samples = (checkpoint_path.parent / "samples.txt").read_text()
            self.assertEqual(samples.count("=== step="), 3)
            for step in (0, 1, 2):
                self.assertIn(f"=== step={step} ", samples)
            self.assertTrue(metadata["generation_config"]["enabled"])
            config.write_text(config.read_text().replace("enabled=true", "enabled=false"))
            with redirect_stdout(io.StringIO()):
                second_checkpoint = train_from_config(config)
            self.assertFalse((second_checkpoint.parent / "samples.txt").exists())
            disabled, _, _ = load_checkpoint(second_checkpoint.parent, device="cpu")
            self.assertTrue(mx.array_equal(loaded(probe)[0], disabled(probe)[0]).item())
            self.assertEqual(second_checkpoint.parent.name, "run-02")
            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(second_checkpoint.is_file())

    def test_sequential_memmap_batches_cover_every_target_once(self):
        import numpy as np
        from llm_training.pretraining_mlx.data import sequential_batch
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokens.bin"
            np.arange(17, dtype="<u4").tofile(path)
            tokens = np.memmap(path, dtype="<u4", mode="r")
            inputs, targets = [], []
            for step in range(2):
                x, y = sequential_batch(tokens, 2, 4, step)
                self.assertEqual(x.shape, (2, 4))
                inputs.extend(x.reshape(-1).tolist())
                targets.extend(y.reshape(-1).tolist())
            self.assertEqual(inputs, list(range(16)))
            self.assertEqual(targets, list(range(1, 17)))
            with self.assertRaises(ValueError):
                sequential_batch(tokens, 2, 4, 2)
            x, y = get_batch(tokens, 2, 4, random.Random(42))
            self.assertTrue(mx.array_equal(y, x + 1).item())

    @patch("llm_training.tokenization.data.iter_dataset_texts")
    def test_cached_sequential_config_training_needs_no_data_fetch(self, texts):
        from llm_training.pretraining.token_cache import prepare_cache
        texts.side_effect = lambda **kwargs: (text for text in ["hello world " * 20])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = train_byte_level_bpe(["hello world"], vocab_size=300, show_progress=False)
            tokenizer.save(str(root / "tokenizer.json"))
            source = dict(dataset_id="local", subset="local", train_split="train", val_split="test",
                          train_documents=20_000_000, val_documents=1, shuffle_buffer_size=0)
            prepare_cache(root / "cache", source, root / "tokenizer.json", "<|endoftext|>", 17)
            config = root / "config.toml"
            config.write_text(
                '[data]\ndataset_id="local"\nsubset="local"\ntrain_split="train"\n'
                'val_split="test"\ntrain_documents=20000000\nval_documents=1\nshuffle_buffer_size=0\n'
                f'[tokenizer]\npath="{root}/tokenizer.json"\neos_token="<|endoftext|>"\n'
                '[model]\ncontext_length=4\nn_layers=1\nn_heads=1\nd_model=8\n'
                '[training]\nsteps=2\nbatch_size=2\nsampling="sequential"\neval_batches=1\ndevice="cpu"\n'
                '[generation]\nenabled=false\n'
                f'[cache]\ndirectory="{root}/cache"\n[output]\ndirectory="{root}/run"\n')
            texts.side_effect = AssertionError("Training must use the local cache")
            with redirect_stdout(io.StringIO()):
                checkpoint = train_from_config(config)
            self.assertEqual(json.loads(checkpoint.read_text())["step"], 2)
            metadata = json.loads((checkpoint.parent / "metadata.json").read_text())
            self.assertEqual(metadata["token_cache"]["train"]["tokens"], 17)
            self.assertEqual(metadata["training_config"]["sampling"], "sequential")
            config.write_text(config.read_text().replace("steps=2", "steps=3"))
            with self.assertRaisesRegex(ValueError, "sequential steps"):
                train_from_config(config)

    def test_generation_eos_and_invalid_inputs(self):
        from unittest.mock import Mock
        from llm_training.pretraining_mlx.generate import generate_text
        tokenizer = train_byte_level_bpe(["hello"], vocab_size=300, show_progress=False)
        eos = tokenizer.token_to_id("<|endoftext|>")
        model = Mock()
        model.generate.side_effect = lambda tokens, **kwargs: mx.concatenate(
            (tokens, mx.array([[eos]], dtype=mx.int32)), axis=1)
        text = generate_text(model, tokenizer, "hello", max_new_tokens=10, eos_id=eos)
        self.assertEqual(text, "hello<|endoftext|>")
        model.generate.assert_called_once()
        for kwargs in ({"prompt": ""}, {"temperature": 0}, {"temperature": float("nan")},
                       {"max_new_tokens": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_text(model, tokenizer, **({"prompt": "hello"} | kwargs))

    def test_periodic_samples_cadence_rng_mode_and_failure_flush(self):
        from llm_training.pretraining_mlx.sampling import GenerationConfig, log_samples
        model = GPT(GPTConfig(vocab_size=300, context_length=4, n_layers=1, n_heads=1, d_model=8))
        mx.eval(model.parameters())
        tokenizer = train_byte_level_bpe(["hello"], vocab_size=300, show_progress=False)
        eos = tokenizer.token_to_id("<|endoftext|>")
        config = GenerationConfig(every=3, prompts=["hello"], max_new_tokens=2)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            mx.random.seed(7)
            expected = mx.random.uniform(shape=(4,))
            mx.eval(expected)
            mx.random.seed(7)
            with log_samples(root, model, tokenizer, eos, config, 5) as record:
                for step in range(6):
                    record(step)
                self.assertTrue(model.training)
                self.assertTrue(mx.array_equal(mx.random.uniform(shape=(4,)), expected).item())
                text = (root / "samples.txt").read_text()
                self.assertEqual(text.count("=== step="), 3)
                for step in (0, 3, 5):
                    self.assertIn(f"=== step={step} ", text)
                with patch("llm_training.pretraining_mlx.sampling.generate_text",
                           side_effect=RuntimeError("sampling failed")):
                    with self.assertRaises(RuntimeError):
                        record(3)
                self.assertTrue(model.training)
                self.assertEqual((root / "samples.txt").read_text(), text)
        for kwargs in ({"every": 0}, {"prompts": []}, {"temperature": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GenerationConfig(**kwargs)

    def test_evaluation_keeps_training_rng_and_mode(self):
        from llm_training.pretraining_mlx.train import evaluate
        model = GPT(GPTConfig(vocab_size=4, context_length=4, n_layers=1, n_heads=1, d_model=8))
        tokens = mx.array([0, 1, 2, 3] * 5)
        config = TrainConfig(eval_batches=2, batch_size=2, device="cpu")
        mx.random.seed(7)
        expected = mx.random.uniform(shape=(4,))
        mx.eval(expected)
        mx.random.seed(7)
        first = evaluate(model, tokens, config)
        actual = mx.random.uniform(shape=(4,))
        self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertEqual(first, evaluate(model, tokens, config))
        self.assertTrue(model.training)

    def test_adamw_decays_only_matrices(self):
        from llm_training.pretraining_mlx.train import MatrixAdamW
        optimizer = MatrixAdamW(learning_rate=0.1, weight_decay=0.2, bias_correction=True)
        parameters = {"matrix": mx.ones((2, 2)), "bias": mx.ones((2,))}
        updated = optimizer.apply_gradients(
            {key: mx.zeros_like(value) for key, value in parameters.items()}, parameters)
        self.assertTrue(mx.allclose(updated["matrix"], mx.full((2, 2), 0.98)).item())
        self.assertTrue(mx.array_equal(updated["bias"], parameters["bias"]).item())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
    def test_matches_pytorch_forward_and_optimizer_update(self):
        import numpy as np
        import torch
        from llm_training.models.gpt import GPT as TorchGPT, GPTConfig as TorchConfig
        from llm_training.pretraining_mlx.train import MatrixAdamW
        torch.set_num_threads(1)
        dimensions = dict(vocab_size=8, context_length=4, n_layers=1, n_heads=2, d_model=8)
        reference = TorchGPT(TorchConfig(**dimensions))
        model = GPT(GPTConfig(**dimensions))
        weights = dict(reference.named_parameters())
        # Sequential layer names differ; linear weight layouts already agree.
        for name, parameter in tree_flatten(model.parameters()):
            source = weights[name.replace("mlp.layers.", "mlp.")]
            self.assertEqual(parameter.shape, tuple(source.shape))
        model.load_weights([(name, mx.array(weights[name.replace("mlp.layers.", "mlp.")]
                                           .detach().numpy()))
                            for name, _ in tree_flatten(model.parameters())])
        ids = [[0, 1, 2, 3], [3, 2, 1, 0]]
        targets = [[1, 2, 3, 4], [2, 1, 0, 7]]
        logits, loss = model(mx.array(ids), mx.array(targets))
        reference_logits, reference_loss = reference(torch.tensor(ids), torch.tensor(targets))
        np.testing.assert_allclose(np.array(logits), reference_logits.detach().numpy(), atol=2e-6)
        self.assertAlmostEqual(loss.item(), reference_loss.item(), places=5)
        optimizer = MatrixAdamW(learning_rate=0.001, betas=[0.9, 0.95],
                                weight_decay=0.1, bias_correction=True)
        reference_optimizer = torch.optim.AdamW([
            {"params": [p for p in reference.parameters() if p.ndim >= 2], "weight_decay": 0.1},
            {"params": [p for p in reference.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ], lr=0.001, betas=(0.9, 0.95))
        for _ in range(2):
            _, gradients = nn.value_and_grad(
                model, lambda m: m(mx.array(ids), mx.array(targets))[1])(model)
            reference_optimizer.zero_grad()
            reference(torch.tensor(ids), torch.tensor(targets))[1].backward()
            gradients, _ = optim.clip_grad_norm(
                gradients, 1.0)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            reference_optimizer.step()
        for name, parameter in tree_flatten(model.parameters()):
            np.testing.assert_allclose(np.array(parameter),
                                       weights[name.replace("mlp.layers.", "mlp.")].detach().numpy(),
                                       atol=2e-5, err_msg=name)

    @unittest.skipUnless(mx.metal.is_available(), "Metal GPU unavailable")
    def test_tiny_metal_training_step(self):
        mx.set_default_device(mx.gpu)
        try:
            model = GPT(GPTConfig(vocab_size=4, context_length=4, n_layers=1, n_heads=1, d_model=8))
            tokens = mx.array([0, 1, 2, 3] * 5)
            with redirect_stdout(io.StringIO()):
                train(model, tokens, tokens, TrainConfig(steps=1, batch_size=1,
                      eval_batches=1, device="gpu"))
            self.assertTrue(mx.isfinite(model(tokens[None, :4], tokens[None, 1:5])[1]).item())
        finally:
            mx.set_default_device(mx.cpu)

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
