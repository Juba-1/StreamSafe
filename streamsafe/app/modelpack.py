"""The model pack: models are not inside the installer; they are put into ``<data>/models`` on the first run.

Four ways in, one way to trust:
  * ``download_from_base`` the files listed in the ``manifest.json`` found at ``release.MODEL_BASE_URL``,
  * ``install_zip``     a zip of the same files that the streamer picked (we sent it to them),
  * ``install_folder``  the same files, already unpacked in a folder the streamer picked,
  * ``install_files``   single files named by the signed update manifest (``updates.py``).
The risk heads are not part of a pack: they ship inside the installer and are updated by ``heads.py``.
Whatever arrives is first assembled in a private folder beside the model folder and checked there exactly as the
engine checks at every start (``onnx_backend.checked_files``: size and sha256 of every file against
``manifest.json``, the pinned model revisions) plus: every file is listed in the manifest, and the risk heads
load. Only then is it moved into place in one step. The models that were in use are kept as ``models.prev``
(``rollback``). A failed check deletes the new files and leaves the old ones untouched: the engine never
starts with models nobody verified (fail closed).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from streamsafe.app import fetch, unpack
from streamsafe.models.onnx_backend import STAMPS

MANIFEST = "manifest.json"
HEADS = "heads"  # a pack made by an older tool may hold the heads: they are skipped, ``heads.py`` owns them
MAX_PACK_BYTES = 16 << 30
MAX_PACK_FILES = 64
ALLOWED_SUFFIXES = (".onnx", ".data", ".npy", ".json", ".txt", ".md")
MANIFEST_MAX_BYTES = 1 << 20
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
NEED_FREE_FACTOR = 1.1

Progress = Callable[[str, int, int], None]  # (what is happening, bytes done, bytes in all)


class PackError(Exception):
    """The model files offered are not a complete, undamaged model pack. Nothing was changed."""


class SourceError(PackError):
    """The place the models should come from did not give them. ``kind``: "private" (401 / 403: the repository is
    not public yet or needs a login), "missing" (404), "network" (no connection, timeout), "other"."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def source_error(exc: fetch.DownloadError) -> SourceError:
    kind = "private" if exc.status in (401, 403) else "missing" if exc.status in (404, 410) else "network" if not exc.status else "other"
    return SourceError(kind, str(exc))


def check_name(name: str) -> str:
    """A pack file name: one plain file name with a known suffix (no folders). Returns it, or raises PackError."""
    if not (_NAME.match(name) and name.lower().endswith(ALLOWED_SUFFIXES)):
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
    return ""


def verify(folder: Path) -> dict:
    """The full check of an assembled pack (module docstring). Returns the manifest's files, or raises PackError."""
    from streamsafe.config import Config
    from streamsafe.models.onnx_backend import checked_files

    try:
        files = checked_files(Config.from_env({}, onnx_dir=str(folder)))
        kept = (MANIFEST, STAMPS, STAMPS + ".tmp")  # the digests kept by the check itself are not part of the pack
        present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name not in kept}
        unlisted = sorted(present - set(files))
        if unlisted:
            raise PackError(f"files that the manifest does not list: {', '.join(unlisted)}")
        for name in files:
            check_name(name)
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
            name = name[len(strip) :]
            return None if name.startswith((HEADS + "/", "trt_cache/")) else check_name(name)

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
        names = [
            check_name(p.relative_to(source).as_posix())
            for p in sorted(source.rglob("*"))
            if p.is_file() and p.name != STAMPS and not {"trt_cache", HEADS} & set(p.relative_to(source).parts[:-1])
        ]
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


def download_from_base(
    base_url: str,
    models: Path,
    downloads: Path,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    allow_http: bool = False,
) -> list[str]:
    """Download ``manifest.json`` from ``base_url`` and every file it lists, check each against the manifest,
    install them in one step. Raises SourceError when the place does not give the files (private / missing /
    no network), PackError when what arrived is not a good pack."""
    base = base_url.rstrip("/") + "/"
    try:
        raw = fetch.fetch_small(base + MANIFEST, MANIFEST_MAX_BYTES, allow_http)
    except fetch.DownloadError as exc:
        raise source_error(exc) from exc
    try:
        files = json.loads(raw.decode("utf-8"))["files"]
        wanted = {MANIFEST: {"url": base + MANIFEST, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}
        for name, meta in files.items():
            wanted[check_name(str(name))] = {"url": base + str(name), "sha256": str(meta["sha256"]).lower(), "bytes": int(meta["bytes"])}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PackError(f"the model list at the download address is not readable ({exc})") from exc
    try:
        return install_files(wanted, models, downloads, progress, cancel, allow_http)
    except OSError as exc:
        raise PackError(str(exc)) from exc


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
    current = current_shas(models)
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
                    part = fetch.download(
                        str(meta.get("url", "")),
                        downloads / f"update-{meta['sha256'][:16]}",
                        meta["sha256"],
                        size,
                        step,
                        cancel,
                        allow_http,
                    )
                except fetch.CancelledError:
                    raise
                except fetch.DownloadError as exc:
                    raise SourceError(source_error(exc).kind, f"{name}: {exc}") from exc
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


def current_shas(models: Path) -> dict[str, str]:
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
