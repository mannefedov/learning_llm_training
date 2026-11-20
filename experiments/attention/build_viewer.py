"""Write a local launch page for the attention canvas app."""

from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "artifacts/analysis/attention-sinks/attention-explorer.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(Path(__file__).with_name("attention_viewer.html").read_text(), encoding="utf-8")
    print(output)
    print("Start the viewer with .venv/bin/python experiments/attention/serve_viewer.py")


if __name__ == "__main__":
    main()
