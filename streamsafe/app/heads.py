"""The risk heads: the small files that turn the eye's embedding into "cover / do not cover". They are ours, not
downloaded from the model repository: the installer ships them (``<program>/heads``).

An update (signed manifest, ``updates.py``) may bring newer heads. They are kept in ``<data>/heads``; the engine uses
whichever of the two folders holds the higher ``heads.json`` version, the installed ones on a tie. The folder in
use is checked like the models: sha256 of every file against the signed manifest, then loaded as the engine loads
them. A folder that does not load is never used; if neither does, nothing starts (fail closed).

``PLACEHOLDER`` in a heads folder marks the synthetic heads a test build of the installer carries (CI only):
they load, but they decide nothing real, so a stream is never started with them.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from streamsafe.app import fetch, paths

META = "heads.json"
PLACEHOLDER = "PLACEHOLDER"
FILES = ("fast_h6.txt", "male_h6.txt", "thr_h6.txt")
OPTIONAL = ("scene_h.txt",)
ALLOWED = (*FILES, *OPTIONAL, META)
MAX_BYTES = 8 << 20

Progress = Callable[[int, int], None]


class HeadsError(Exception):
    """The head files offered are not usable. Nothing was changed."""


def version(folder: Path) -> int:
    """The ``version`` in ``heads.json`` (0 when there is none)."""
    try:
        return int(json.loads((folder / META).read_text(encoding="utf-8"))["version"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def problem(folder: Path) -> str:
    """"" when ``folder`` holds heads the engine can load, otherwise why not."""
    from streamsafe.models.heads import RiskHeads

    if (folder / PLACEHOLDER).exists():
        return "placeholder"
    try:
        RiskHeads.load(folder)
    except FileNotFoundError:
        return "missing"
    except (OSError, ValueError, IndexError) as exc:
        return f"damaged ({exc})"
    return ""


def best(env: Mapping[str, str] | None = None, app_folder: Path | None = None) -> Path | None:
    """The heads folder the engine should use, or None when there is no usable one."""
    shipped = app_folder if app_folder is not None else paths.bundled_heads_dir()
    updated = paths.heads_dir(env)
    usable = [(version(f), i, f) for i, f in enumerate((shipped, updated)) if not problem(f)]
    # higher version first; on a tie the installed one (index 0)
    return min(usable, key=lambda t: (-t[0], t[1]))[2] if usable else None


def why_not(env: Mapping[str, str] | None = None, app_folder: Path | None = None) -> str:
    """"" when some heads can be used, otherwise ``missing`` / ``placeholder`` / the reason."""
    if best(env, app_folder) is not None:
        return ""
    shipped = app_folder if app_folder is not None else paths.bundled_heads_dir()
    return problem(shipped) or "missing"


def previous(folder: Path) -> Path:
    return folder.with_name(folder.name + ".prev")


def install_files(
    wanted: dict[str, dict],
    new_version: int,
    folder: Path,
    downloads: Path,
    cancel: threading.Event | None = None,
    allow_http: bool = False,
) -> list[str]:
    """Bring ``folder`` to exactly ``wanted`` (file name -> {"url", "sha256", "bytes"}; from the signed manifest).
    Each file is downloaded and its sha256 checked, the set is loaded as the engine would load it, and only then
    moved into place; the folder in use becomes ``<folder>.prev``. Returns the names downloaded."""
    names = set(wanted)
    if not set(FILES) <= names or not names <= set(ALLOWED) - {META}:
        raise HeadsError("the update's head files are not the expected ones")
    stage = folder.with_name(f"{folder.name}.new-{os.getpid()}-{int(time.time())}")
    for old in folder.parent.glob(f"{folder.name}.new-*"):
        shutil.rmtree(old, ignore_errors=True)
    folder.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    try:
        for name, meta in sorted(wanted.items()):
            size = int(meta.get("bytes", 0))
            if not 0 < size <= MAX_BYTES:
                raise HeadsError(f"the update's entry for {name} is incomplete")
            try:
                part = fetch.download(
                    str(meta.get("url", "")), downloads / f"head-{str(meta.get('sha256', ''))[:16]}", str(meta.get("sha256", "")), size, None, cancel, allow_http
                )
            except fetch.Cancelled:
                raise
            except fetch.DownloadError as exc:
                raise HeadsError(f"{name}: {exc}") from exc
            shutil.copyfile(part, stage / name)
            part.unlink(missing_ok=True)
        (stage / META).write_text(json.dumps({"version": int(new_version)}), encoding="utf-8")
        if why := problem(stage):
            raise HeadsError(f"the new heads do not load: {why}")
        prev = previous(folder)
        if folder.exists():
            shutil.rmtree(prev, ignore_errors=True)
            os.replace(folder, prev)
        try:
            os.replace(stage, folder)
        except OSError:
            if prev.exists() and not folder.exists():
                os.replace(prev, folder)
            raise
    except OSError as exc:
        raise HeadsError(str(exc)) from exc
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return sorted(names)


def rollback(folder: Path) -> None:
    """Swap the updated heads with the ones kept from before the last update."""
    prev = previous(folder)
    if not prev.is_dir() or problem(prev):
        raise HeadsError("there are no earlier heads to go back to")
    aside = folder.with_name(f"{folder.name}.swap-{os.getpid()}")
    try:
        if folder.exists():
            os.replace(folder, aside)
        os.replace(prev, folder)
        if aside.exists():
            os.replace(aside, prev)
    except OSError as exc:
        raise HeadsError(str(exc)) from exc
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(aside, ignore_errors=True)
