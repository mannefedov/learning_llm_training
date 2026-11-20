"""Serve saved attention probes and their canvas viewer on localhost."""

import argparse
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


def make_handler(data_dir, static_dir):
    def catalog():
        summaries = json.loads((data_dir / "summary.json").read_text())
        result = []
        for index, summary in enumerate(summaries):
            run = summary["run"]
            if Path(run).name != run:
                raise ValueError("Invalid run name in probe summary")
            path = data_dir / f"{run}.npz"
            with np.load(path) as source:
                shape = source["attention"].shape
            if len(shape) != 4 or shape[-1] != shape[-2]:
                raise ValueError(f"Invalid attention shape for {run}")
            result.append({**summary, "id": index, "shape": shape,
                           "version": path.stat().st_mtime_ns})
        return result

    @lru_cache(maxsize=8)
    def map_bytes(run, version):
        with np.load(data_dir / f"{run}.npz") as source:
            values = source["attention"]
        if not np.isfinite(values).all():
            raise ValueError("Probe contains non-finite attention probabilities")
        return values.astype("<f4").tobytes()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def send_content(self, content, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlparse(self.path).path
            if not path.startswith("/api/"):
                return super().do_GET()
            try:
                entries = catalog()
                if path == "/api/checkpoints":
                    self.send_content(json.dumps(entries).encode(), "application/json")
                elif path.startswith("/api/maps/"):
                    index = int(path.removeprefix("/api/maps/"))
                    if index < 0 or index >= len(entries):
                        raise ValueError("Unknown checkpoint")
                    entry = entries[index]
                    self.send_content(map_bytes(entry["run"], entry["version"]),
                                      "application/octet-stream")
                else:
                    self.send_error(404)
            except (ValueError, KeyError, OSError) as error:
                self.send_error(400, str(error))

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path,
                        default=Path("artifacts/analysis/attention-sinks"))
    args = parser.parse_args()
    static = Path(__file__).with_name("viewer")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.data.resolve(), static))
    print(f"Attention viewer at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
