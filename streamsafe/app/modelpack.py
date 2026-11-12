"""The model pack: models are not inside the installer; they are put into ``<data>/models`` on the first run.

Three ways in, one way to trust:
  * ``install_zip``     a zip made by ``tools/make_model_pack.py`` (downloaded from ``release.MODEL_PACK_URL`` or
                        picked by the streamer),
  * ``install_folder``  the same files, already unpacked in a folder the streamer picked,
  * ``install_files``   single files named by the signed update manifest (``updates.py``).
Whatever arrives is first assembled in a private folder beside the model folder and checked there exactly as the
engine checks at every start (``onnx_backend.checked_files``: size and sha256 of every file against
``manifest.json``, the pinned model revisions) plus: every file is listed in the manifest, and the risk heads
load. Only then is it moved into place in one step. The models that were in use are kept as ``models.prev``
(``rollback``). A failed check deletes the new files and leaves the old ones untouched: the engine never
starts with models nobody verified (fail closed).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from streamsafe.app import fetch, unpack

MANIFEST = "manifest.json"
HEADS = "heads"
MAX_PACK_BYTES = 16 << 30
MAX_PACK_FILES = 64
ALLOWED_SUFFIXES = (".onnx", ".data", ".npy", ".json", ".txt", ".md")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
NEED_FREE_FACTOR = 1.1

Progress = Callable[[str, int, int], None]  # (what is happening, bytes done, bytes in all)


class PackError(Exception):
    """The model files offered are not a complete, undamaged model pack. Nothing was changed."""


def check_name(name: str) -> str:
    """A pack file name: a plain file, or one inside ``heads/``. Returns it, or raises PackError."""
    parts = name.split("/")
    ok = len(parts) in (1, 2) and all(_NAME.match(p) for p in parts) and name.lower().endswith(ALLOWED_SUFFIXES)
    if not ok or (len(parts) == 2 and parts[0] != HEADS):
        raise PackError(f"not a model pack file: {name!r}")
    return name


def problem(models: Path, quick: bool = True) -> str:
    """ "" when ``models`` holds a usable pack, otherwise why not, in words. ``quick``: sizes only (the window);
    the engine itself hashes every file at every start."""
    try:
        if quick:
            from streamsafe.models.onnx_backend import REQUIRED

            files = json.loads((models / MANIFEST).read_text(encoding="utf-8"))["files"]
            bad = [n for n in REQUIRED if n not in files]
            bad += [n for n, m in files.items() if not (models / n).is_file() or (models / n).stat().st_size != m["bytes"]]
            if bad:
                return f"missing or damaged: {', '.join(sorted(set(bad)))}"
        else:
            verify(models)
    except FileNotFoundError:
        return "the models are not installed yet"
    except (OSError, ValueError, KeyError, TypeError, PackError) as exc:
        return str(exc)
    return "" if (models / HEADS).is_dir() else "the risk heads are missing"


def verify(folder: Path) -> dict:
    """The full check of an assembled pack (module docstring). Returns the manifest's files, or raises PackError."""
    from streamsafe.config import Config
    from streamsafe.models.heads import RiskHeads
    from streamsafe.models.onnx_backend import checked_files

    try:
        files = checked_files(Config.from_env({}, onnx_dir=str(folder)))
        present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} - {MANIFEST}
        unlisted = sorted(present - set(files))
        if unlisted:
            raise PackError(f"files that the manifest does not list: {', '.join(unlisted)}")
        for name in files:
            check_name(name)
        RiskHeads.load(folder / HEADS)
    except PackError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise PackError(str(exc)) from exc
    return files


def _stage(models: Path) -> Path:
    models.parent.mkdir(parents=True, exist_ok=True)
    for old in models.parent.glob(f"{models.name}.new-*"):  # left by an install that was interrupted
        with contextlib.suppress(OSError):
            if time.time() - old.stat().st_mtime > 3600:
                shutil.rmtree(old, ignore_errors=True)
    stage = models.with_name(f"{models.name}.new-{os.getpid()}-{int(time.time())}")
    stage.mkdir()
    return stage


def _need_space(folder: Path, need: int) -> None:
    free = shutil.disk_usage(folder).free
    if free < need * NEED_FREE_FACTOR:
        raise PackError(f"not enough free disk space: {need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")


def previous(models: Path) -> Path:
    return models.with_name(models.name + ".prev")


def _commit(stage: Path, models: Path) -> None:
    """Verify the staged pack and move it into place; the pack in use becomes ``models.prev``."""
    try:
        verify(stage)
        prev = previous(models)
        if models.exists():
            shutil.rmtree(prev, ignore_errors=True)
            os.replace(models, prev)
        try:
            os.replace(stage, models)
        except OSError:
            if prev.exists() and not models.exists():
                os.replace(prev, models)  # put the old pack back: never leave the PC without models
            raise
    except OSError as exc:
        raise PackError(f"the models could not be put in place ({exc}); close the stream and try again") from exc
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def rollback(models: Path) -> None:
    """Swap the pack in use with the one kept from before the last update (verified again first)."""
    prev = previous(models)
    if not prev.is_dir():
        raise PackError("there are no earlier models to go back to")
    verify(prev)
    aside = models.with_name(f"{models.name}.swap-{os.getpid()}")
    try:
        if models.exists():
            os.replace(models, aside)
        os.replace(prev, models)
        if aside.exists():
            os.replace(aside, prev)
    except OSError as exc:
        raise PackError(f"could not go back to the earlier models ({exc})") from exc


def install_zip(
    archive: Path, models: Path, sha256: str = "", progress: Progress | None = None, cancel: threading.Event | None = None
) -> None:
    """Install a pack zip. ``sha256`` (from the release settings) is checked first when given."""
    if sha256:
        _tell(progress, "verifying")
        if fetch.sha256_file(archive) != sha256.lower():
            raise PackError("this is not the model pack of this version (sha256 mismatch)")
    stage = _stage(models)
    try:
        import zipfile

        try:
            with zipfile.ZipFile(archive) as zf:
                names = [i.filename for i in zf.infolist() if not i.is_dir()]
                need = sum(i.file_size for i in zf.infolist())
        except (zipfile.BadZipFile, OSError) as exc:
            raise PackError(f"not a readable zip file ({exc})") from exc
        # a pack zipped as one folder ("models/manifest.json") is accepted: that folder is dropped
        tops = {n.split("/", 1)[0] for n in names}
        strip = f"{next(iter(tops))}/" if len(tops) == 1 and MANIFEST not in names and all("/" in n for n in names) else ""
        _need_space(stage, need)

        def choose(name: str) -> str | None:
            return check_name(name[len(strip) :])

        try:
            unpack.extract(archive, stage, choose, MAX_PACK_BYTES, MAX_PACK_FILES, _named(progress, "unpacking"), cancel)
        except (unpack.UnsafeArchiveError, OSError) as exc:
            raise PackError(str(exc)) from exc
        _tell(progress, "verifying")
        _commit(stage, models)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def install_folder(source: Path, models: Path, progress: Progress | None = None, cancel: threading.Event | None = None) -> None:
    """Install a pack that is already unpacked in ``source`` (copied, so the source stays as it is)."""
    if not (source / MANIFEST).is_file():
        raise PackError(f"this folder has no {MANIFEST}: it is not a model pack")
    try:
        names = [check_name(p.relative_to(source).as_posix()) for p in sorted(source.rglob("*")) if p.is_file() and "trt_cache" not in p.parts]
        total = sum((source / n).stat().st_size for n in names)
    except OSError as exc:
        raise PackError(str(exc)) from exc
    if len(names) > MAX_PACK_FILES or total > MAX_PACK_BYTES:
        raise PackError("this folder holds more than a model pack")
    stage = _stage(models)
    try:
        _need_space(stage, total)
        done = 0
        try:
            for name in names:
                out = stage / name
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(source / name, "rb") as src, open(out, "wb") as dst:
                    while chunk := src.read(unpack.CHUNK):
                        if cancel is not None and cancel.is_set():
                            raise InterruptedError("cancelled")
                        dst.write(chunk)
                        done += len(chunk)
                        _tell(progress, "copying", done, total)
        except OSError as exc:
            raise PackError(str(exc)) from exc
        _tell(progress, "verifying")
        _commit(stage, models)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def download_and_install(
    url: str,
    sha256: str,
    models: Path,
    downloads: Path,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    allow_http: bool = False,
) -> None:
    """Download the pack zip (pinned sha256 required), install it, delete the zip."""
    archive = fetch.download(url, downloads / "model_pack.zip", sha256, 0, _named(progress, "downloading"), cancel, allow_http, MAX_PACK_BYTES)
    try:
        install_zip(archive, models, sha256, progress, cancel)
    finally:
        archive.unlink(missing_ok=True)


def install_files(
    wanted: dict[str, dict],
    models: Path,
    downloads: Path,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    allow_http: bool = False,
) -> list[str]:
    """Bring the model folder to exactly ``wanted`` (file name -> {"url", "sha256", "bytes"}, from the signed
    update manifest; it must name ``manifest.json`` too). Files already here with the same sha256 are copied
    from the pack in use, the others are downloaded. Returns the names that were downloaded."""
    if MANIFEST not in wanted:
        raise PackError(f"the update does not name {MANIFEST}")
    for name, meta in wanted.items():
        if name != MANIFEST:
            check_name(name)
        if not (isinstance(meta, dict) and re.fullmatch(r"[0-9a-f]{64}", str(meta.get("sha256", ""))) and int(meta.get("bytes", 0)) > 0):
            raise PackError(f"the update's entry for {name} is incomplete")
    total = sum(int(m["bytes"]) for m in wanted.values())
    if len(wanted) > MAX_PACK_FILES or total > MAX_PACK_BYTES:
        raise PackError("the update names more than a model pack")
    current = _current_shas(models)
    stage = _stage(models)
    fetched, done = [], 0
    try:
        _need_space(stage, total)
        for name, meta in sorted(wanted.items()):
            out = stage / name
            out.parent.mkdir(parents=True, exist_ok=True)
            size = int(meta["bytes"])
            if current.get(name) == meta["sha256"] and (models / name).is_file() and (models / name).stat().st_size == size:
                shutil.copyfile(models / name, out)  # verified again with everything else before the swap
            else:
                base = done

                def step(got: int, _all: int, base: int = base) -> None:
                    _tell(progress, "downloading", base + got, total)

                try:
                    part = fetch.download(str(meta.get("url", "")), downloads / f"update-{meta['sha256'][:16]}", meta["sha256"], size, step, cancel, allow_http)
                except fetch.Cancelled:
                    raise
                except fetch.DownloadError as exc:
                    raise PackError(f"{name}: {exc}") from exc
                os.replace(part, out)
                fetched.append(name)
            done += size
            _tell(progress, "downloading", done, total)
        _tell(progress, "verifying")
        _commit(stage, models)
    except OSError as exc:
        raise PackError(str(exc)) from exc
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return fetched


def _current_shas(models: Path) -> dict[str, str]:
    """sha256 by name of the pack in use, as its manifest says (trusted only as a hint: all is verified again)."""
    try:
        files = json.loads((models / MANIFEST).read_text(encoding="utf-8"))["files"]
        out = {str(n): str(m["sha256"]) for n, m in files.items()}
        out[MANIFEST] = fetch.sha256_file(models / MANIFEST)
        return out
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _tell(progress: Progress | None, what: str, done: int = 0, total: int = 0) -> None:
    if progress:
        progress(what, done, total)


def _named(progress: Progress | None, what: str) -> Callable[[int, int], None] | None:
    return (lambda done, total: progress(what, done, total)) if progress else None
