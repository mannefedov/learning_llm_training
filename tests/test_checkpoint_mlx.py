"""Tiny checkpoint retention, interrupted-save, and resume equivalence checks."""

import csv
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

from llm_training.pretraining_mlx.__main__ import tomllib, train_from_config
from llm_training.pretraining_mlx.checkpoint import CheckpointConfig, CheckpointWriter, checkpoint_tokens
from llm_training.pretraining_mlx.data import sequential_batch
from llm_training.pretraining.token_cache import prepare_cache
from llm_training.tokenization.bpe import train_byte_level_bpe


class CheckpointTests(unittest.TestCase):
    @patch("llm_training.tokenization.data.iter_dataset_texts",
           side_effect=lambda **kwargs: (s for s in ["hello world " * 50]))
    def test_stable_checkpoint_survives_decay_and_resumes_without_decay(self, texts):
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = self.make_config(root, steps=7)
            config.write_text(config.read_text().replace(
                '[training]\n', '[training]\nlr_schedule="cosine"\nwarmup_steps=1\n'
                'lr_decay_start_step=3\nlr_decay_steps=7\n'))
            pointer = train_from_config(config)
            stable = pointer.parent / "stable-checkpoint.json"
            state = json.loads(stable.read_text())
            self.assertEqual(state["step"], 3)  # Forced save between periodic steps 2 and 4.
            self.assertEqual(sorted(p.name for p in (pointer.parent / "checkpoints").iterdir()),
                             ["step-000000003", "step-000000007"])
            self.assertTrue((stable.parent / state["optimizer"]).is_file())
            config.write_text(config.read_text().replace("steps=7", "steps=9").replace(
                '[training]\n', '[training]\nlr_decay=false\n'))
            resumed = train_from_config(config, resume=stable)
            metadata = json.loads((resumed.parent / "metadata.json").read_text())
            self.assertEqual(metadata["start_step"], 3)
            self.assertEqual(metadata["data_start_step"], 0)
            self.assertFalse(metadata["training_config"]["lr_decay"])
            final = json.loads(resumed.read_text())
            stable_final = json.loads((resumed.parent / "stable-checkpoint.json").read_text())
            self.assertEqual(stable_final["step"], final["step"])
            with (resumed.parent / "metrics.csv").open() as stream:
                rates = [float(row["learning_rate"]) for row in csv.DictReader(stream)
                         if row["learning_rate"]]
            self.assertTrue(all(abs(rate - .0003) < 1e-10 for rate in rates))
            # Explicit resume annealing must override the disabled-decay setting.
            annealed = train_from_config(config, resume=resumed, anneal_steps=2)
            self.assertTrue(json.loads(annealed.read_text())["training_config"]["lr_decay"])

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
            'attention_types=["sliding_window", "full"]\nsliding_window=2\nstable_softmax=true\n'
            f'[training]\nsteps={steps}\nbatch_size=2\nsampling="{sampling}"\n'
            'eval_every=2\neval_batches=1\ndevice="cpu"\ncompile_step=true\n'
            f'[checkpoint]\nevery=2\nkeep={keep}\n[generation]\nenabled=false\n'
            f'[cache]\ndirectory="{root}/cache"\n[output]\ndirectory="{root}/run"\n')
        return config

    @patch("llm_training.tokenization.data.iter_dataset_texts",
           side_effect=lambda **kwargs: (s for s in ["hello world " * 50]))
    def test_anneal_uses_saved_lr_and_resumes_on_same_curve(self, texts):
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = self.make_config(root)
            config.write_text(config.read_text().replace(
                '[training]\n', '[training]\nlr_schedule="cosine"\nmin_learning_rate=0.00004\n'
                'warmup_steps=1\n'))
            source = train_from_config(config)
            saved = json.loads(source.read_text())
            saved_lr = mx.load(str(source.parent / saved["optimizer"]))["learning_rate"].item()
            self.assertAlmostEqual(saved_lr, 0.00004)
            config.write_text(config.read_text().replace("eval_every=2", "eval_every=1"))
            branch = train_from_config(config, resume=source, anneal_steps=5)
            with (branch.parent / "metrics.csv").open() as metrics:
                rows = list(csv.DictReader(metrics))
            self.assertEqual([int(row["step"]) for row in rows], [4, 5, 6, 7, 8, 9])
            self.assertEqual(float(rows[1]["learning_rate"]), saved_lr)
            self.assertAlmostEqual(float(rows[3]["learning_rate"]), saved_lr / 2)
            self.assertEqual(float(rows[-1]["learning_rate"]), 0.0)

            save = CheckpointWriter.__call__

            def interrupt(writer, step, optimizer, generator):
                save(writer, step, optimizer, generator)
                if step == 7:
                    raise RuntimeError("interrupted")

            with patch.object(CheckpointWriter, "__call__", interrupt):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    train_from_config(config, resume=source, anneal_steps=5)
            interrupted = branch.parent.with_name("run-anneal-5-02")
            resumed = train_from_config(interrupted / "config.toml", resume=interrupted)
            for field in ("model", "optimizer"):
                expected = mx.load(str(branch.parent / json.loads(branch.read_text())[field]))
                actual = mx.load(str(resumed.parent / json.loads(resumed.read_text())[field]))
                self.assertEqual(set(actual), set(expected))
                self.assertTrue(all(mx.array_equal(actual[key], expected[key]).item() for key in actual))

    @patch("llm_training.tokenization.data.iter_dataset_texts",
           side_effect=lambda **kwargs: (s for s in ["hello world " * 50, "fresh tokens " * 50]))
    def test_new_sequential_cache_starts_at_zero_and_anneal_keeps_position(self, texts):
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = self.make_config(root)
            source = train_from_config(config)
            next_cache = root / "next-cache"
            source_state = json.loads(source.read_text())
            self.assertEqual(checkpoint_tokens(source_state), 32)
            source_state.pop("tokens_processed_total")
            self.assertEqual(checkpoint_tokens(source_state), 32)  # Legacy checkpoint.
            next_text = config.read_text().replace(
                "[data]\n", "[data]\ntrain_skip_documents=1\n").replace(
                "steps=4", "steps=6").replace('directory="' + str(root / "cache") + '"',
                                              'directory="' + str(next_cache) + '"')
            config.write_text(next_text)
            settings = tomllib.loads(next_text)
            prepare_cache(next_cache, settings["data"], root / "tokenizer.json",
                          "<|endoftext|>", 41, after_cache=root / "cache")
            self.assertEqual((next_cache / "val.bin").read_bytes(),
                             (root / "cache/val.bin").read_bytes())
            with self.assertRaisesRegex(ValueError, "data/cache differs"):
                train_from_config(config, resume=source)
            with patch("llm_training.pretraining_mlx.train.sequential_batch",
                       wraps=sequential_batch) as batches:
                direct = train_from_config(config, resume=source, new_cache=True, anneal_steps=2)
                self.assertEqual([call.args[-1] for call in batches.call_args_list], [0, 1])
                self.assertEqual(json.loads(direct.read_text())["data_start_step"], 4)
                batches.reset_mock()
                continuation = train_from_config(config, resume=source, new_cache=True)
                self.assertEqual([call.args[-1] for call in batches.call_args_list], [0, 1])
                batches.reset_mock()
                branch = train_from_config(config, resume=continuation, anneal_steps=2)
                self.assertEqual([call.args[-1] for call in batches.call_args_list], [2, 3])
            self.assertEqual(json.loads(branch.read_text())["data_start_step"], 4)
            self.assertEqual(checkpoint_tokens(json.loads(branch.read_text())), 64)
            config.write_text(next_text.replace("[training]\n",
                                                "[training]\ngradient_accumulation_steps=2\n"))
            with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                train_from_config(config, resume=source, new_cache=True, anneal_steps=2)
            with patch("llm_training.pretraining_mlx.train.sequential_batch",
                       wraps=sequential_batch) as batches:
                larger = train_from_config(config, resume=source, new_cache=True,
                                           anneal_steps=2, allow_accumulation_change=True)
                self.assertEqual([call.args[-1] for call in batches.call_args_list], [0, 1])
                self.assertEqual([call.args[1] for call in batches.call_args_list], [4, 4])
            larger_meta = json.loads((larger.parent / "metadata.json").read_text())
            self.assertEqual(larger_meta["effective_batch_size"], 4)
            self.assertEqual(json.loads(larger.read_text())["data_start_step"], 4)
            self.assertEqual(checkpoint_tokens(json.loads(larger.read_text())), 64)
            config.write_text(config.read_text().replace("steps=6", "steps=8"))
            followup = train_from_config(config, resume=larger)
            self.assertEqual(checkpoint_tokens(json.loads(followup.read_text())), 96)

    @patch("llm_training.tokenization.data.iter_dataset_texts",
           side_effect=lambda **kwargs: (s for s in ["hello world " * 50]))
    def test_staircase_branch_boundaries_and_interrupted_resume(self, texts):
        from llm_training.pretraining_mlx.train import TrainConfig, learning_rate_at_step

        schedule = TrainConfig(steps=12, lr_schedule="staircase", learning_rate=1.0,
                               min_learning_rate=0, lr_decay_start_step=5, lr_decay_steps=12)
        self.assertEqual([learning_rate_at_step(schedule, s) for s in range(5, 13)],
                         [1, 1, .5, .5, .25, .25, .1, 0])
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = self.make_config(root)
            source = train_from_config(config)
            full = train_from_config(config, resume=source, anneal_steps=8,
                                     anneal_schedule="staircase")
            save = CheckpointWriter.__call__

            def interrupt(writer, step, optimizer, generator):
                save(writer, step, optimizer, generator)
                if step == 8:
                    raise RuntimeError("interrupted")

            with patch.object(CheckpointWriter, "__call__", interrupt):
                with self.assertRaises(RuntimeError):
                    train_from_config(config, resume=source, anneal_steps=8,
                                      anneal_schedule="staircase")
            branch = root / "run-anneal-8-02"
            resumed = train_from_config(branch / "config.toml", resume=branch)
            for field in ("model", "optimizer"):
                expected = mx.load(str(full.parent / json.loads(full.read_text())[field]))
                actual = mx.load(str(resumed.parent / json.loads(resumed.read_text())[field]))
                self.assertTrue(all(mx.array_equal(actual[key], expected[key]).item() for key in actual))

    @patch("llm_training.tokenization.data.iter_dataset_texts",
           side_effect=lambda **kwargs: (s for s in ["hello world " * 50]))
    def test_interrupted_run_resumes_identically_for_both_sampling_modes(self, texts):
        for sampling, schedule, accumulation in (("random", "constant", 1), ("sequential", "constant", 1),
                                                 ("sequential", "cosine", 1), ("sequential", "cosine", 2),
                                                 ("sequential", "stable", 2),
                                                 ("random", "cosine", 2)):
            with self.subTest(sampling=sampling, schedule=schedule, accumulation=accumulation), TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
                root = Path(temporary)
                config = self.make_config(root, sampling, steps=8)
                config.write_text(config.read_text().replace(
                    '[training]\n', f'[training]\ngradient_accumulation_steps={accumulation}\n'))
                if schedule in ("cosine", "stable"):
                    config.write_text(config.read_text().replace(
                        '[training]\n', '[training]\nlr_schedule="cosine"\n'
                        'warmup_steps=2\nlr_decay_steps=8\nmin_learning_rate=0.00003\n'))
                    if schedule == "stable":
                        config.write_text(config.read_text().replace(
                            '[training]\n', '[training]\nlr_decay=false\n'))
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
                if accumulation == 1:
                    # Checkpoints from before accumulation imply factor one.
                    saved = json.loads(pointer.read_text())
                    saved["training_config"].pop("gradient_accumulation_steps")
                    pointer.write_text(json.dumps(saved))
                self.assertEqual(len(list((root / "run/checkpoints").glob("step-*"))),
                                 2 if schedule == "cosine" else 1)
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
                self.assertEqual(metadata["effective_batch_size"], 2 * accumulation)
                self.assertEqual(metadata["tokens_per_update"], 8 * accumulation)
                if sampling == "sequential":
                    self.assertEqual(metadata["steps_per_epoch"], 5 // accumulation)
                with self.assertRaisesRegex(ValueError, "total greater"):
                    train_from_config(config, resume=resumed)
                original_config = config.read_text()
                config.write_text(original_config.replace("compile_step=true", "compile_step=false"))
                toggled = train_from_config(config, resume=pointer)
                toggled_meta = json.loads((toggled.parent / "metadata.json").read_text())
                self.assertFalse(toggled_meta["training_config"]["compile_step"])
                toggled_state = json.loads(toggled.read_text())
                actual = mx.load(str(toggled.parent / toggled_state["model"]))
                expected = mx.load(str(baseline.parent / baseline_state["model"]))
                for name in actual:
                    self.assertTrue(mx.allclose(actual[name], expected[name], atol=2e-6, rtol=2e-5).item(), name)
                config.write_text(original_config.replace("stable_softmax=true", "stable_softmax=false"))
                with self.assertRaisesRegex(ValueError, "model configuration"):
                    train_from_config(config, resume=pointer)
                config.write_text(original_config)
                config.write_text(config.read_text().replace(
                    f"gradient_accumulation_steps={accumulation}", f"gradient_accumulation_steps={accumulation + 1}"))
                with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                    train_from_config(config, resume=pointer)
                config.write_text(original_config)
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
