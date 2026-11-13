"""Check the preparation CLI's success-only native-shutdown workaround."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class PrepareCliTests(unittest.TestCase):
    def test_success_bypasses_shutdown_but_errors_keep_normal_exit(self):
        for fail in (False, True):
            with self.subTest(fail=fail), TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = root / "config.toml"
                marker = root / "atexit.txt"
                config.write_text(
                    '[data]\n[tokenizer]\npath="unused.json"\neos_token="eos"\n'
                    '[training]\nsteps=1\nbatch_size=1\n[model]\ncontext_length=4\n'
                    '[cache]\ndirectory="unused-cache"\n'
                )
                script = '''
import atexit
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

config, marker, fail = sys.argv[1:]
atexit.register(lambda: Path(marker).write_text("normal shutdown"))
sys.argv = ["prepare", "--config", config]
error = RuntimeError("preparation failed") if fail == "True" else None
with patch("llm_training.pretraining.token_cache.prepare_cache", side_effect=error):
    runpy.run_module("llm_training.pretraining_mlx.prepare", run_name="__main__")
'''
                result = subprocess.run([sys.executable, "-c", script, str(config), str(marker), str(fail)],
                                        capture_output=True, text=True, timeout=20)
                if fail:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("preparation failed", result.stderr)
                    self.assertNotIn("Cache ready", result.stdout)
                    self.assertTrue(marker.exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Cache ready: unused-cache", result.stdout)
                    self.assertFalse(marker.exists())
