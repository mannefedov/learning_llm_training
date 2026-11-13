"""Deterministic progress timing and terminal/log output checks."""

from contextlib import redirect_stdout
import io
import os
import unittest
from unittest.mock import patch

from llm_training.pretraining.progress import TrainingProgress, format_duration


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def test_elapsed_eta_percent_and_terminal_cleanup(self):
        output = Terminal()
        with redirect_stdout(output), patch(
            "llm_training.pretraining.progress.time.monotonic", return_value=0
        ) as clock, patch(
            "llm_training.pretraining.progress.shutil.get_terminal_size", return_value=os.terminal_size((160, 30))
        ):
            with TrainingProgress(10) as progress:
                clock.return_value = 20
                progress.update(2, train_loss=3.5, val_loss=4.0)
                self.assertIn("20.0% 2/10 elapsed=00:00:20 ETA=00:01:20", output.getvalue())
                clock.return_value = 100
                progress.update(10)
        self.assertIn("finished", output.getvalue())
        self.assertIn("100.0%", output.getvalue())
        self.assertIn("train=3.5000 val=4.0000", output.getvalue())
        self.assertTrue(output.getvalue().endswith("\n"))
        self.assertIn("\r", output.getvalue())

    def test_redirected_logs_are_sparse_and_interrupts_keep_actual_step(self):
        output = io.StringIO()
        with redirect_stdout(output), patch(
            "llm_training.pretraining.progress.time.monotonic", return_value=0
        ) as clock:
            with self.assertRaises(KeyboardInterrupt):
                with TrainingProgress(10) as progress:
                    clock.return_value = 5
                    progress.update(1)
                    self.assertEqual(len(output.getvalue().splitlines()), 1)
                    clock.return_value = 30
                    progress.update(2)
                    self.assertEqual(len(output.getvalue().splitlines()), 2)
                    raise KeyboardInterrupt
        self.assertIn("stopped", output.getvalue())
        self.assertNotIn("100.0%", output.getvalue())
        self.assertNotIn("\r", output.getvalue())

    def test_duration(self):
        self.assertEqual(format_duration(3661.9), "01:01:01")
        self.assertEqual(format_duration(0), "00:00:00")
