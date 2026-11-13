"""Small terminal progress display shared by both training backends."""

import sys
import shutil
import time


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgress:
    """Refresh one terminal line; emit sparse lines when output is redirected.

    ETA uses average wall time per completed update, including validation and
    sampling. It excludes data preparation and final checkpoint writing.
    """

    def __init__(self, total: int):
        self.total = total
        self.step = 0
        self.train_loss = None
        self.val_loss = None
        self.stream = sys.stdout
        self.interactive = self.stream.isatty()
        self.started = time.monotonic()
        self.last_display = self.started
        self.line_width = 0

    def __enter__(self):
        self._display()
        return self

    def update(self, step: int, train_loss=None, val_loss=None):
        self.step = step
        if train_loss is not None:
            self.train_loss = train_loss
        if val_loss is not None:
            self.val_loss = val_loss
        interval = 1.0 if self.interactive else 30.0
        if time.monotonic() - self.last_display >= interval:
            self._display()

    def _display(self, status="training"):
        now = time.monotonic()
        elapsed = now - self.started
        eta = format_duration(elapsed / self.step * (self.total - self.step)) if self.step else "--:--:--"
        filled = int(16 * self.step / self.total)
        bar = "#" * filled + "-" * (16 - filled)
        losses = ""
        if self.train_loss is not None:
            losses += f" train={self.train_loss:.4f}"
        if self.val_loss is not None:
            losses += f" val={self.val_loss:.4f}"
        line = (f"{status} [{bar}] {100 * self.step / self.total:5.1f}% "
                f"{self.step}/{self.total} elapsed={format_duration(elapsed)} ETA={eta}{losses}")
        if self.interactive:
            # Avoid wrapping an updating line on narrow terminals.
            width = max(1, shutil.get_terminal_size().columns - 1)
            if len(line) > width:
                line = line.replace(f" [{bar}]", "")
            line = line[:width]
            self.stream.write("\r" + line.ljust(min(self.line_width, width)))
            self.line_width = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last_display = now

    def __exit__(self, error_type, error, traceback):
        self._display("finished" if error_type is None else "stopped")
        if self.interactive:
            self.stream.write("\n")
            self.stream.flush()
