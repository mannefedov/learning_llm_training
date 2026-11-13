"""Offline checks for collision-safe output allocation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from llm_training.runs import create_run_directory


class RunDirectoryTests(unittest.TestCase):
    def test_numbered_paths_skip_existing_files_and_directories(self):
        with TemporaryDirectory() as temporary:
            requested = Path(temporary) / "nested" / "run-01"
            self.assertEqual(create_run_directory(requested), requested)
            requested.with_name("run-02").write_text("keep")
            self.assertEqual(create_run_directory(requested).name, "run-03")
            self.assertEqual(requested.with_name("run-02").read_text(), "keep")

    def test_unnumbered_paths_and_number_width_overflow(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, expected in (("tokenizer", "tokenizer-02"), ("run-99", "run-100")):
                requested = root / name
                create_run_directory(requested)
                self.assertEqual(create_run_directory(requested).name, expected)
