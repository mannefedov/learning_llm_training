"""Small terminal progress display shared by both training backends."""

import os
import shutil
import sys
import textwrap
import time


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgress:
    """Refresh a terminal status block; emit sparse lines for redirected output.

    ETA uses average wall time per completed update, including validation and
    sampling. It excludes data preparation and final checkpoint writing.
    """

    def __init__(self, total: int, start_step: int = 0, steps_per_epoch: int | None = None,
                 tokens_per_step: int = 0, parameter_count: int | None = None,
                 tokens_before_start: int | None = None):
        self.tokens_per_step = tokens_per_step
        self.parameter_count = parameter_count
        self.tokens_before_start = (start_step * tokens_per_step
                                    if tokens_before_start is None else tokens_before_start)
        self.total = total
        self.steps_per_epoch = steps_per_epoch
        self.step = start_step
        self.start_step = start_step
        self.train_loss = None
        self.val_loss = None
        self.learning_rate = None
        self.stream = sys.stdout
        self.interactive = self.stream.isatty() and os.environ.get("TERM") != "dumb"
        self.display_rows = 0
        self.started = time.monotonic()
        self.last_display = self.started

    def __enter__(self):
        self._display()
        return self

    def update(self, step: int, train_loss=None, val_loss=None, learning_rate=None):
        self.step = step
        if train_loss is not None:
            self.train_loss = train_loss
        if val_loss is not None:
            self.val_loss = val_loss
        if learning_rate is not None:
            self.learning_rate = learning_rate
        interval = 1.0 if self.interactive else 30.0
        if time.monotonic() - self.last_display >= interval:
            self._display()

    def metrics(self):
        """Wall-clock throughput for this invocation; count training targets once."""
        elapsed = time.monotonic() - self.started
        tokens = (self.step - self.start_step) * self.tokens_per_step
        return {"elapsed_seconds": elapsed, "tokens_processed": tokens,
                "tokens_per_second": tokens / elapsed if tokens and elapsed > 0 else None}

    @property
    def tokens_per_parameter(self):
        if not self.parameter_count:
            return None
        tokens = self.tokens_before_start + (self.step - self.start_step) * self.tokens_per_step
        return tokens / self.parameter_count

    def _display(self, status="training"):
        now = time.monotonic()
        elapsed = now - self.started
        completed = self.step - self.start_step
        eta = format_duration(elapsed / completed * (self.total - self.step)) if completed else "--:--:--"
        filled = int(16 * self.step / self.total)
        bar = "#" * filled + "-" * (16 - filled)
        losses = ""
        if self.train_loss is not None:
            losses += f" train={self.train_loss:.4f}"
        if self.val_loss is not None:
            losses += f" val={self.val_loss:.4f}"
        if self.learning_rate is not None:
            losses += f" lr={self.learning_rate:.2e}"
        epochs = (f" epochs={self.step / self.steps_per_epoch:.2f}/{self.total / self.steps_per_epoch:.2f}"
                  if self.steps_per_epoch else "")
        rate = ""
        if self.tokens_per_step:
            speed = completed * self.tokens_per_step / elapsed if completed and elapsed > 0 else None
            rate = f" tok/s={speed:,.0f}" if speed is not None else " tok/s=--"
        if self.tokens_per_parameter is not None:
            rate += f" tok/param={self.tokens_per_parameter:.2f}"
        line = (f"{status} [{bar}] {100 * self.step / self.total:5.1f}% "
                f"{self.step}/{self.total}{rate} elapsed={format_duration(elapsed)} ETA={eta}{epochs}{losses}")
        if self.interactive:
            self._redraw(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last_display = now

    def _redraw(self, line):
        try:
            width = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            width = shutil.get_terminal_size().columns
        # Keep a spare column so the terminal doesn't wrap on its own.
        rows = textwrap.wrap(line, width=max(1, width - 1), break_on_hyphens=False)
        # The cursor is on the last row of the previous block. Erase its rows
        # from bottom to top, then write the replacement from that same origin.
        clear = "\r\x1b[2K" + "\x1b[1A\x1b[2K" * max(0, self.display_rows - 1)
        self.stream.write(clear + "\r\n".join(rows))
        self.display_rows = len(rows)

    def __exit__(self, error_type, error, traceback):
        self._display("finished" if error_type is None else "stopped")
        if self.interactive:
            self.stream.write("\n")
            self.stream.flush()
