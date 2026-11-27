#!/usr/bin/env python3
"""Serve site/ locally: revalidated, gzip-compressed, keep-alive.

Plain `python3 -m http.server` sends no Cache-Control (browsers may reuse stale
ES modules after an edit), speaks HTTP/1.0 without keep-alive (a new
connection per file, slow through an SSH tunnel) and never compresses (the
repository JSON shrinks about 5x with gzip).
"""
import argparse
import functools
import gzip
import http.server
import os
import re
from pathlib import Path

COMPRESS = {".json", ".js", ".css", ".html", ".svg", ".txt", ".md"}
_cache: dict[str, tuple[float, bytes]] = {}


class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        # Code is never kept (no-store); data is revalidated (no-cache).
        code = self.path.split("?")[0].endswith((".html", ".js", ".css", "/"))
        self.send_header("Cache-Control", "no-store" if code else "no-cache")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        # Byte ranges, as static hosts serve them: the trace viewer reads one chunk of a pack at a time.
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        if match and os.path.isfile(path) and Path(path).suffix not in COMPRESS:
            size = os.path.getsize(path)
            start = int(match.group(1))
            end = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
            if start > end:
                self.send_error(416)
                return None
            with open(path, "rb") as handle:
                handle.seek(start)
                body = handle.read(end - start + 1)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return _Body(body)
        if (os.path.isfile(path) and Path(path).suffix in COMPRESS
                and "gzip" in self.headers.get("Accept-Encoding", "")):
            stat = os.stat(path)
            modified = self.date_time_string(int(stat.st_mtime))
            if self.headers.get("If-Modified-Since") == modified:
                self.send_response(304)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            cached = _cache.get(path)
            if not cached or cached[0] != stat.st_mtime:
                with open(path, "rb") as handle:
                    cached = (stat.st_mtime, gzip.compress(handle.read(), compresslevel=6))
                _cache[path] = cached
            body = cached[1]
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Last-Modified", modified)
            self.end_headers()
            return _Body(body)
        return super().send_head()


class _Body:
    """A file-like object over bytes, as send_head's contract expects."""

    def __init__(self, data: bytes):
        self.data = data

    def read(self, *_):
        data, self.data = self.data, b""
        return data

    def close(self):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("port", type=int, nargs="?", default=8000)
    parser.add_argument("--bind", default="127.0.0.1")
    options = parser.parse_args()
    site = Path(__file__).resolve().parent / "site"
    handler = functools.partial(Handler, directory=str(site))
    with http.server.ThreadingHTTPServer((options.bind, options.port), handler) as server:
        print(f"Serving {site} at http://{options.bind}:{options.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
