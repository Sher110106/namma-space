#!/usr/bin/env python3
"""Serve a web folder (default: viewer/dist) for local use. Standard library only.

  python3 viewer/tools/serve.py [DIR] [--port 5178] [--cors]

Sends Range support and a correct MIME type for .spz/.js/.json/.png, so it behaves like GitHub Pages for this viewer.
Offline use for judges: unzip the bundle, run `python3 -m http.server` in the folder, open http://127.0.0.1:8000/ .
"""
from __future__ import annotations

import argparse
import functools
import http.server
import mimetypes
import os
import re
import sys
from pathlib import Path

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".spz")


class Handler(http.server.SimpleHTTPRequestHandler):
    cors = False

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if self.cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range, Accept-Ranges")
        super().end_headers()

    def send_head(self):  # add single-range support
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = min(size - 1, int(m.group(2))) if (m.group(1) and m.group(2)) else size - 1
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            buf = source.read(min(65536, left))
            if not buf:
                break
            outputfile.write(buf)
            left -= len(buf)
        self._range_left = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default=str(Path(__file__).resolve().parent.parent / "dist"))
    ap.add_argument("--port", type=int, default=5178)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cors", action="store_true")
    a = ap.parse_args()
    Handler.cors = a.cors
    handler = functools.partial(Handler, directory=a.dir)
    with http.server.ThreadingHTTPServer((a.host, a.port), handler) as srv:
        print(f"serving {a.dir} at http://{a.host}:{a.port}/", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
