"""Helpers for the app tests: a local HTTP server, a fake model pack, a fake protector."""

from __future__ import annotations

import hashlib
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass


class Server:
    """Serves ``root`` on 127.0.0.1; ``status`` forces a status code for every request (401 / 404 ...)."""

    def __init__(self, root: Path, status: int = 0) -> None:
        outer = self

        class Handler(Quiet):
            def do_GET(self):
                if outer.status:
                    self.send_error(outer.status)
                    return
                outer.hits.append(self.path)
                super().do_GET()

        self.status, self.hits = status, []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/"

    def __enter__(self) -> Server:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_pack(folder: Path, payload: bytes = b"model-bytes", names=("detector.onnx", "eye.onnx", "eye.onnx.data")) -> dict:
    """A folder with manifest.json + files, shaped like what tools/export_onnx.py writes."""
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in names:
        data = payload + name.encode()
        (folder / name).write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": sha(data)}
    (folder / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return files


def light_verify(folder: Path) -> dict:
    """Stands in for modelpack.verify in tests that have no real models: every listed file exists with its sha."""
    from streamsafe.app import modelpack

    files = json.loads((folder / "manifest.json").read_text())["files"]
    for name, meta in files.items():
        data = (folder / name).read_bytes()
        if len(data) != meta["bytes"] or sha(data) != meta["sha256"]:
            raise modelpack.PackError(f"{name} damaged")
    present = {p.name for p in folder.iterdir() if p.is_file()} - {"manifest.json"}
    if present - set(files):
        raise modelpack.PackError("unlisted files")
    return files


class FakeProtector:
    """Reversible 'protection' for tests (Linux has no DPAPI). Not encryption."""

    def protect(self, data: bytes) -> bytes:
        return bytes(b ^ 0x5A for b in data)[::-1]

    def unprotect(self, blob: bytes) -> bytes:
        return bytes(b ^ 0x5A for b in blob[::-1])


def make_heads(folder: Path, version: int = 1, n: int = 8, thr: str = "0.5") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    line = f"1000;0;{','.join(['1'] * n)}"
    for name in ("fast_h6.txt", "male_h6.txt"):
        (folder / name).write_text(line, encoding="utf-8")
    (folder / "thr_h6.txt").write_text(thr, encoding="utf-8")
    (folder / "heads.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    return folder
