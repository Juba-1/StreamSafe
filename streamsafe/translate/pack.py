"""The translation pack: the files this feature needs, downloaded the first time it is switched on (not part of
the installer, not part of the main model pack).

  det.onnx / rec.onnx              text detector + English recogniser (PaddleOCR PP-OCRv4 / PP-OCRv3, Apache-2.0)
  model.bin, config.json, shared_vocabulary.json, source.spm, target.spm
                                   Helsinki-NLP/opus-mt-en-ar converted to CTranslate2 int8 (see NOTICE-translate.txt)
  NotoSansArabic-Regular.ttf, NotoSans-Regular.ttf   fonts (SIL OFL 1.1)
  NOTICE-translate.txt, LICENSE-*.txt                attributions and licence texts that must travel with the files

Every file is pinned by size and sha256 in ``release.TRANSLATE_FILES`` (``tools/build_translate_pack.py`` writes
the table); nothing is kept that does not match. A pack is installed in one step from a staging folder, so the
folder in use is never half written.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from streamsafe.app import fetch

MANIFEST = "manifest.json"
REQUIRED = (
    "det.onnx",
    "rec.onnx",
    "model.bin",
    "config.json",
    "shared_vocabulary.json",
    "source.spm",
    "target.spm",
    "NotoSansArabic-Regular.ttf",
    "NotoSans-Regular.ttf",
    "NOTICE-translate.txt",
)
Progress = Callable[[str, int, int], None]


class PackError(Exception):
    """The translation pack is not complete or not the pinned one. Nothing in use was changed."""


def _pins(pins: dict | None) -> dict:
    if pins is not None:
        return pins
    from streamsafe.app import release

    return {n: {"sha256": s, "bytes": b} for n, (s, b) in release.TRANSLATE_FILES.items()}


def problem(folder: str | Path, pins: dict | None = None, full: bool = False) -> str:
    """ "" when ``folder`` holds a usable pack, otherwise why not. The pinned table decides what is right; when no
    table is pinned (a development build) the pack's own manifest does. ``full``: every sha256 is checked."""
    folder = Path(folder) if folder else None
    if folder is None or not folder.is_dir():
        return "the translation files are not installed"
    try:
        want = _pins(pins)
        if not want:
            want = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))["files"]
        missing = [n for n in REQUIRED if n not in want or not (folder / n).is_file()]
        if missing:
            return f"missing: {', '.join(missing)}"
        for n, meta in want.items():
            p = folder / n
            if not p.is_file() or p.stat().st_size != int(meta["bytes"]):
                return f"damaged: {n}"
            if full and fetch.sha256_file(p) != str(meta["sha256"]).lower():
                return f"damaged: {n}"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return str(exc)
    return ""


def _commit(stage: Path, folder: Path) -> None:
    try:
        old = folder.with_name(folder.name + ".old")
        if folder.exists():
            shutil.rmtree(old, ignore_errors=True)
            os.replace(folder, old)
        try:
            os.replace(stage, folder)
        except OSError:
            if old.exists() and not folder.exists():
                os.replace(old, folder)
            raise
        shutil.rmtree(old, ignore_errors=True)
    except OSError as exc:
        raise PackError(f"the translation files could not be put in place ({exc})") from exc


def install_download(
    base_url: str,
    folder: Path,
    downloads: Path,
    pins: dict | None = None,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    allow_http: bool = False,
) -> None:
    """Download every pinned file from ``base_url`` and install them. Raises ``fetch.DownloadError`` (network,
    private, missing) or PackError."""
    want = _pins(pins)
    if not all(n in want for n in REQUIRED):
        raise PackError("no translation files are pinned in this build")
    base = base_url.rstrip("/") + "/"
    total = sum(int(m["bytes"]) for m in want.values())
    stage = folder.with_name(folder.name + f".new-{os.getpid()}")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    done = 0
    try:
        for name, meta in sorted(want.items()):
            base_done = done

            def step(got: int, _all: int, base_done: int = base_done) -> None:
                if progress:
                    progress("downloading", base_done + got, total)

            part = fetch.download(
                base + name, downloads / f"translate-{meta['sha256'][:16]}", meta["sha256"], int(meta["bytes"]), step, cancel, allow_http
            )
            shutil.copyfile(part, stage / name)
            done += int(meta["bytes"])
        (stage / MANIFEST).write_text(json.dumps({"files": want}, indent=1), encoding="utf-8")
        why = problem(stage, want, full=True)
        if why:
            raise PackError(why)
        folder.parent.mkdir(parents=True, exist_ok=True)
        _commit(stage, folder)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def install_folder(source: Path, folder: Path, pins: dict | None = None) -> None:
    """Install a pack that was handed over as a folder (checked against the pinned table first)."""
    if not source.is_dir():
        raise PackError("not a folder")
    why = problem(source, pins, full=True)
    if why:
        raise PackError(why)
    stage = folder.with_name(folder.name + f".new-{os.getpid()}")
    shutil.rmtree(stage, ignore_errors=True)
    try:
        shutil.copytree(source, stage, ignore=shutil.ignore_patterns("*.part"))
        why = problem(stage, pins, full=True)
        if why:
            raise PackError(why)
        _commit(stage, folder)
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(stage, ignore_errors=True)
