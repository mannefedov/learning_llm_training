"""Tiny checkpoint retention, interrupted-save, and resume equivalence checks."""

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("Install the pretraining-mlx extra")

import mlx.core as mx

from llm_training.pretraining_mlx.__main__ import train_from_config
from llm_training.pretraining_mlx.checkpoint import CheckpointConfig, CheckpointWriter
from llm_training.pretraining.token_cache import prepare_cache
from llm_training.tokenization.bpe import train_byte_level_bpe


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self.device)

    def make_config(self, root, sampling="sequential", steps=4, keep=1):
        tokenizer = train_byte_level_bpe(["hello world"], vocab_size=300, show_progress=False)
        tokenizer.save(str(root / "tokenizer.json"))
        source = dict(dataset_id="local", subset="local", train_split="train", val_split="test",
                      train_documents=1, val_documents=1, shuffle_buffer_size=0)
        prepare_cache(root / "cache", source, root / "tokenizer.json", "<|endoftext|>", 41)
        config = root / "train.toml"
        config.write_text(
            '[data]\ndataset_id="local"\nsubset="local"\ntrain_split="train"\nval_split="test"\n'
            'train_documents=1\nval_documents=1\nshuffle_buffer_size=0\n'
            f'[tokenizer]\npath="{root}/tokenizer.json"\neos_token="<|endoftext|>"\n'
            '[model]\ncontext_length=4\nn_layers=2\nn_heads=1\nd_model=8\n'
            'attention_types=["sliding_window", "full"]\nsliding_window=2\n'
            f'[training]\nsteps={steps}\nbatch_size=2\nsampling="{sampling}"\n'
            'eval_every=2\neval_batches=1\ndevice="cpu"\n'
            f'[checkpoint]\nevery=2\nkeep={keep}\n[generation]\nenabled=false\n'
            f'[cache]\ndirectory="{root}/cache"\n[output]\ndirectory="{root}/run"\n')
        return config

    @patch("llm_training.tokenization.data.iter_dataset_texts",
           side_effect=lambda **kwargs: (s for s in ["hello world " * 50]))
    def test_interrupted_run_resumes_identically_for_both_sampling_modes(self, texts):
        for sampling in ("random", "sequential"):
            with self.subTest(sampling=sampling), TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
                root = Path(temporary)
                config = self.make_config(root, sampling, steps=8)
                save = CheckpointWriter.__call__

                def interrupted(writer, step, optimizer, generator):
                    save(writer, step, optimizer, generator)
                    if step == 7:
                        raise RuntimeError("interrupted after checkpoint")

                with patch.object(CheckpointWriter, "__call__", interrupted):
                    with self.assertRaises(RuntimeError):
                        train_from_config(config)
                pointer = root / "run/checkpoint.json"
                self.assertEqual(json.loads(pointer.read_text())["step"], 6)
                self.assertEqual(len(list((root / "run/checkpoints").glob("step-*"))), 1)
                resumed = train_from_config(config, resume=pointer.parent)
                self.assertEqual(len(list((resumed.parent / "checkpoints").glob("step-*"))), 1)
                baseline = train_from_config(config)
                resumed_state = json.loads(resumed.read_text())
                baseline_state = json.loads(baseline.read_text())
                for field in ("model", "optimizer"):
                    actual = mx.load(str(resumed.parent / resumed_state[field]))
                    expected = mx.load(str(baseline.parent / baseline_state[field]))
                    self.assertEqual(set(actual), set(expected))
                    for name in actual:
                        self.assertTrue(mx.array_equal(actual[name], expected[name]).item(), (field, name))
                metadata = json.loads((resumed.parent / "metadata.json").read_text())
                self.assertEqual(metadata["start_step"], 6)
                with self.assertRaisesRegex(ValueError, "total greater"):
                    train_from_config(config, resume=resumed)
                config.write_text(config.read_text().replace("batch_size=2", "batch_size=1"))
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    train_from_config(config, resume=pointer)

    @patch("llm_training.tokenization.data.iter_dataset_texts",
           side_effect=lambda **kwargs: (s for s in ["hello world " * 50]))
    def test_retention_and_failed_save_preserve_previous_checkpoint(self, texts):
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = self.make_config(root, steps=5, keep=2)
            pointer = train_from_config(config)
            folders = sorted(p.name for p in (pointer.parent / "checkpoints").iterdir())
            self.assertEqual(folders, ["step-000000004", "step-000000005"])
            from llm_training.pretraining_mlx.generate import load_checkpoint
            model, _, _ = load_checkpoint(pointer.parent / "checkpoints/step-000000004", "cpu")
            self.assertEqual(model.config.attention_types, ["sliding_window", "full"])

            # Fail during serialization, before a replacement checkpoint is published.
            config.write_text(config.read_text().replace("steps=5", "steps=4").replace("keep=2", "keep=1"))
            original = mx.save_safetensors
            calls = []

            def fail_second(path, *args, **kwargs):
                if Path(path).name == "optimizer.safetensors":
                    calls.append(path)
                    if len(calls) == 2:
                        raise OSError("disk full")
                return original(path, *args, **kwargs)

            with patch("llm_training.pretraining_mlx.checkpoint.mx.save_safetensors", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "disk full"):
                    train_from_config(config)
            failed = root / "run-02"
            self.assertEqual(json.loads((failed / "checkpoint.json").read_text())["step"], 2)
            self.assertEqual([p.name for p in (failed / "checkpoints").iterdir()], ["step-000000002"])
            config.write_text(config.read_text().replace("[checkpoint]\nevery=2", "[checkpoint]\nevery=0"))
            final_only = train_from_config(config)
            self.assertEqual([p.name for p in (final_only.parent / "checkpoints").iterdir()],
                             ["step-000000004"])
        for kwargs in ({"keep": 0}, {"every": -1}, {"keep": True}):
            with self.assertRaises(ValueError):
                CheckpointConfig(**kwargs)
