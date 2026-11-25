"""The TensorRT engine cache: which engine may be loaded, and how one gets there.

An engine is the model compiled by TensorRT for ONE graphics card model, driver and TensorRT version. It is built
on the streamer's PC, so no manifest shipped with the product can cover it. Two things must never happen:

  * a STALE engine is loaded: built from other model bytes, for another card or driver, or by other libraries.
    ONNX Runtime finds an engine by the model's file name alone, so this module decides, not ONNX Runtime;
  * a PARTIAL or damaged engine is loaded: the build was interrupted (minutes long: the PC is switched off, the
    installer is closed), or the file was cut short or changed on disk afterwards.

How:
  * the KEY is everything the engine depends on (``engine_key``): sha256 of the model file and its weight file,
    name and driver version of every NVIDIA card, ONNX Runtime version, TensorRT version, and the build options
    (fp16, batch range). The engine folder is named after the key, so another key never finds this folder;
  * an engine is BUILT in a private folder beside the final one (``begin``). Only after the build succeeded and
    the new engine answered a test input is the folder SEALED (``seal.json``: the full key + size and sha256 of
    every file) and moved into place in one step (``commit``). An interrupted build leaves a ``.building`` folder
    that nothing ever reads;
  * before loading, ``check`` compares the seal with the key of this start and re-hashes the files. Anything
    that does not match is not loaded: the engine is built again, or CUDA runs the model and the run report says
    why (``session.open_session``).

The folder is trusted like the program itself: it must be writable by the streamer's own user only (the sha256 in
the seal detects damage, not somebody who can also rewrite the seal).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import secrets
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SEAL = "seal.json"
FORMAT = 1  # of the key and the seal: raise it when either changes meaning, old engines are then built again
BUILDING = ".building-"
STALE_BUILD_S = 3600.0  # a ``.building`` folder older than this belongs to a build that was killed (a build takes minutes)
MOVE_TRIES, MOVE_WAIT_S = 10, 0.5  # Windows: a virus scanner may hold the new engine file open for a moment
# pip distributions that carry the TensorRT libraries (``pip install tensorrt-cu12``), most specific first
TRT_DISTS = ("tensorrt-cu12-libs", "tensorrt-cu13-libs", "tensorrt-libs", "tensorrt-cu12", "tensorrt")
TRT_LIBRARY = re.compile(r"^(nvinfer_\d+\.dll|libnvinfer\.so\.\d+)$")  # the core library, Windows / Linux
_KEY_ID = re.compile(r"-[0-9a-f]{16}($|\.building-)")


class NotUsableError(RuntimeError):
    """TensorRT cannot be used on this PC as it is (the message says why, for the report and the streamer)."""


@dataclass(frozen=True)
class Identity:
    """What an engine is tied to on this PC, besides the model itself."""

    gpus: tuple[tuple[str, str], ...]  # (name, driver version) of every NVIDIA card, sorted
    ort: str
    trt: str


def identity(libs: str = "") -> Identity:
    """This PC's identity for the engine key. Raises ``NotUsableError`` when a part of it cannot be read: an engine
    whose card or libraries are unknown could not be told from a stale one, so none is built or loaded."""
    import onnxruntime as ort

    from streamsafe.syscheck import nvidia_gpus

    gpus = tuple(sorted((g.name, g.driver) for g in nvidia_gpus()))
    if not gpus or not all(name and driver for name, driver in gpus):
        raise NotUsableError("the NVIDIA card and its driver version could not be read (nvidia-smi)")
    trt = trt_version(libs)
    if not trt:
        raise NotUsableError("the TensorRT libraries were not found (NVIDIA edition: reinstall; developers: pip install tensorrt-cu12)")
    return Identity(gpus, ort.__version__, trt)


def trt_version(libs: str = "") -> str:
    """Which TensorRT this process will load, "" when none is found.

    ``libs`` (the ``TRT_LIBS`` folder of an installed product) wins: the core library there is named with its size,
    which changes with every release. Otherwise the pip package's version, otherwise the library on the system
    search path."""
    if libs:
        return _library_stamp([libs])
    for dist in TRT_DISTS:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            return f"{dist} {importlib.metadata.version(dist)}"
    return _library_stamp(os.environ.get("PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH", "").split(os.pathsep))


def _library_stamp(folders: list[str]) -> str:
    for folder in filter(None, folders):
        with contextlib.suppress(OSError):
            for path in sorted(Path(folder).iterdir()):
                if TRT_LIBRARY.match(path.name.lower()) and path.is_file():
                    return f"{path.name} {path.stat().st_size} bytes"
    return ""


_EXPOSED: set[str] = set()


def expose_libraries(libs: str = "") -> None:
    """Make the TensorRT libraries loadable by ONNX Runtime without the streamer (or the installer) editing PATH.

    ONNX Runtime's own preloading covers CUDA and cuDNN only. NVIDIA's pip package brings its loader
    (``import tensorrt_libs`` loads every library in its folder); for a ``TRT_LIBS`` folder the same is done here.
    A library that fails to load is not an error at this point: the session reports that TensorRT did not start."""
    if libs in _EXPOSED:
        return
    _EXPOSED.add(libs)
    if not libs:
        if importlib.util.find_spec("tensorrt_libs") is not None:
            with contextlib.suppress(Exception):
                importlib.import_module("tensorrt_libs")
        return
    import ctypes

    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            os.add_dll_directory(libs)
        os.environ["PATH"] = libs + os.pathsep + os.environ.get("PATH", "")
    for path in sorted(Path(libs).glob("*.dll" if sys.platform == "win32" else "*.so*")):
        with contextlib.suppress(OSError):
            ctypes.CDLL(str(path))


def model_shas(path: Path, files: dict) -> dict[str, str]:
    """sha256 of a model file and of its weight file (``eye.onnx`` + ``eye.onnx.data``). ``files`` is the verified
    manifest; a model outside a manifest (the speed bench on a loose file) is hashed here, so the key always
    follows the bytes."""
    from streamsafe.models.onnx_backend import sha256

    out = {name: meta["sha256"] for name, meta in files.items() if name == path.name or name == path.name + ".data"}
    if path.name not in out:
        out = {p.name: sha256(p) for p in (path, path.with_name(path.name + ".data")) if p.exists()}
    return out


def engine_key(model: str, shas: dict[str, str], options: dict, ident: Identity) -> dict:
    """Everything one engine depends on. ``options``: the TensorRT build options (fp16, batch range ...)."""
    return {
        "format": FORMAT,
        "model": model,
        "sha256": dict(sorted(shas.items())),
        "options": {k: options[k] for k in sorted(options)},
        "gpus": [list(g) for g in ident.gpus],
        "ort": ident.ort,
        "trt": ident.trt,
    }


def folder(cache: str | Path, model: str, key: dict) -> Path:
    tag = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return Path(cache) / f"{Path(model).stem}-{tag}"


def listing(path: Path) -> dict[str, tuple[int, int]]:
    """name -> (size, time written) of every file in an engine folder except the seal and the kept digests."""
    from streamsafe.models.onnx_backend import STAMPS

    out = {}
    for p in sorted(path.iterdir()):
        if p.name not in (SEAL, STAMPS, STAMPS + ".tmp"):
            st = p.stat()
            out[p.name] = (st.st_size, st.st_mtime_ns)
    return out


def check(path: Path, key: dict, quick: bool = False) -> str:
    """ "" when ``path`` holds a complete engine built for exactly ``key``, otherwise why it must not be loaded.
    ``quick``: sizes only, no sha256 (the system checker); otherwise every file's sha256, re-hashed only when the
    file changed since it was last hashed (``sha256_known``)."""
    from streamsafe.models.onnx_backend import sha256_known

    if not path.is_dir():
        return "not built yet"
    try:
        seal = json.loads((path / SEAL).read_text())
        sealed, files = seal["key"], {str(name): (int(meta["bytes"]), str(meta["sha256"])) for name, meta in seal["files"].items()}
    except FileNotFoundError:
        return "the engine folder has no seal (incomplete)"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return "the engine's seal is unreadable"
    if sealed != key:
        return explain(sealed, key)
    try:
        present = listing(path)
    except OSError as exc:
        return f"the engine folder cannot be read ({exc})"
    if set(present) != set(files) or not any(name.endswith(".engine") for name in files):
        return "the engine folder does not hold the files that were sealed"
    for name, (size, digest) in files.items():
        if present[name][0] != size or (not quick and sha256_known(path / name) != digest):
            return f"{name} is damaged (size / sha256 mismatch)"
    return ""


def explain(old: object, new: dict) -> str:
    """Why an engine sealed with ``old`` does not fit ``new``, in words a streamer can act on."""
    if not isinstance(old, dict):
        return "the engine's seal is unreadable"
    old_gpus, new_gpus = old.get("gpus") or [], new["gpus"]
    if [g[0] for g in old_gpus] != [g[0] for g in new_gpus]:
        return f"the graphics card changed ({_names(old_gpus, 0)} -> {_names(new_gpus, 0)})"
    if old_gpus != new_gpus:
        return f"the graphics driver changed ({_names(old_gpus, 1)} -> {_names(new_gpus, 1)})"
    for part, text in (("trt", "TensorRT"), ("ort", "ONNX Runtime")):
        if old.get(part) != new[part]:
            return f"{text} changed ({old.get(part)} -> {new[part]})"
    if old.get("sha256") != new["sha256"]:
        return "the model file changed"
    return "the engine settings changed"


def _names(gpus: list, index: int) -> str:
    return ", ".join(str(g[index]) for g in gpus) or "none"


def why_not_ready(cache: str | Path, model: str, key: dict, quick: bool = False) -> str:
    """``check`` for the engine of ``key``; when it was never built, the reason names what changed since the
    engine that IS in the cache was built (new driver, new card ...), if there is one."""
    why = check(folder(cache, model, key), key, quick)
    if why != "not built yet":
        return why
    for other in _siblings(Path(cache), model):
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            return explain(json.loads((other / SEAL).read_text())["key"], key)
    return why


def _siblings(cache: Path, model: str) -> list[Path]:
    """Sealed engine folders of the same model under other keys, newest first."""
    stem = Path(model).stem
    found = [p for p in cache.glob(f"{stem}-*") if _KEY_ID.search(p.name[len(stem) :]) and BUILDING not in p.name and (p / SEAL).exists()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def begin(final: Path) -> Path:
    """A private, empty folder to build into. It is never read by anybody until ``commit`` moves it into place."""
    final.parent.mkdir(parents=True, exist_ok=True)
    work = final.with_name(f"{final.name}{BUILDING}{os.getpid()}-{secrets.token_hex(4)}")
    work.mkdir()
    return work


def seal(path: Path, key: dict) -> None:
    """Write the seal: the last step of a build. Raises OSError when TensorRT left no engine behind."""
    from streamsafe.models.onnx_backend import sha256

    files = {name: {"bytes": size, "sha256": sha256(path / name)} for name, (size, _) in listing(path).items()}
    if not any(name.endswith(".engine") for name in files):
        raise OSError("TensorRT wrote no engine file (disk full, or it took no part of the model)")
    part = path / (SEAL + ".part")
    part.write_text(json.dumps({"key": key, "files": files}, indent=1))
    os.replace(part, path / SEAL)


def commit(work: Path, final: Path, key: dict) -> str:
    """Seal the finished build and move it into place. "" when stored; otherwise why it could not be kept (the
    engine in memory is still good for this run, it is just built again next time)."""
    try:
        seal(work, key)
        for attempt in range(MOVE_TRIES):
            if final.exists():
                # another process finished the same engine first: keep theirs. The full check, with sha256: an
                # engine damaged at the same size must not be mistaken for that and survive its own rebuild
                if not check(final, key):
                    remove(work)
                    return ""
                remove(final)  # stale or damaged: that is why this build ran
            try:
                os.replace(work, final)
                return ""
            except OSError:
                if attempt == MOVE_TRIES - 1:
                    raise
                time.sleep(MOVE_WAIT_S)
    except OSError as exc:
        remove(work)
        return f"the engine could not be stored ({exc})"
    return ""


def remove(path: Path) -> None:
    """Delete one engine folder (stale, damaged, or an abandoned build). Best effort: what cannot be deleted is
    still never loaded, because ``check`` fails for it."""
    if _KEY_ID.search(path.name):  # only ever folders this module named
        shutil.rmtree(path, ignore_errors=True)


def sweep(cache: str | Path, keep: list[Path]) -> list[str]:
    """After a successful prepare: delete the engines of ``keep``'s models built under other keys (an old driver,
    an old model: gigabytes nobody will load again) and builds that were killed. Returns the names removed."""
    cache, removed = Path(cache), []
    for current in keep:
        stem = current.name.rsplit("-", 1)[0]
        for p in cache.glob(f"{stem}-*"):
            if p == current or not _KEY_ID.search(p.name[len(stem) :]):
                continue
            with contextlib.suppress(OSError):
                if BUILDING in p.name and time.time() - p.stat().st_mtime < STALE_BUILD_S:
                    continue  # may be another process building right now
                remove(p)
                removed.append(p.name)
    return sorted(removed)
