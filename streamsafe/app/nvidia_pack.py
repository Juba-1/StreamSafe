"""The fast mode for NVIDIA cards: NVIDIA's own libraries, downloaded on the streamer's PC from NVIDIA's package
indexes after the streamer said yes. We never ship them (docs/NVIDIA_TRT.md, section 1: the licence question).

What is fetched is the list in ``release.NVIDIA_WHEELS``: ONNX Runtime's GPU build (Microsoft), the CUDA and
cuDNN runtimes and TensorRT (NVIDIA), each a wheel (a zip) pinned by sha256. They are unpacked side by side
into ``<data>/nvidia/site`` like a Python ``site-packages`` folder, which is what ``runtime_select`` puts in
front of the bundled DirectML runtime when the engine is started in NVIDIA mode.

All or nothing: the folder is assembled beside the final one and moved into place only when every wheel was
verified and unpacked; ``READY.json`` in it names the pins it was built from. Anything less is not "ready" and
the app stays on DirectML and says so.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from streamsafe.app import fetch, release, unpack

READY = "READY.json"
MAX_WHEEL_UNPACKED = 8 << 30

Progress = Callable[[str, int, int], None]  # (wheel name, bytes done of all downloads, bytes in all or 0)


class NvidiaPackError(Exception):
    """The NVIDIA libraries could not be installed. Nothing was changed; DirectML stays in use."""


def site_dir(nvidia: Path) -> Path:
    return nvidia / "site"


def _pins(wheels: tuple[release.Wheel, ...]) -> dict[str, str]:
    return {w.name: w.sha256 for w in wheels}


def ready(nvidia: Path, wheels: tuple[release.Wheel, ...] = release.NVIDIA_WHEELS) -> bool:
    """Is a complete set, unpacked from exactly this build's pins, in place?"""
    site = site_dir(nvidia)
    try:
        sealed = json.loads((site / READY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return sealed == {"wheels": _pins(wheels)} and (site / "onnxruntime" / "__init__.py").is_file()


def dll_dirs(nvidia: Path) -> list[Path]:
    """The folders that hold the DLLs ONNX Runtime's CUDA and TensorRT providers load."""
    site = site_dir(nvidia)
    found = [site / "onnxruntime" / "capi", site / "tensorrt_libs"]
    found += sorted(p for p in (site / "nvidia").glob("*/bin") if p.is_dir())
    return [p for p in found if p.is_dir()]


def install(
    nvidia: Path,
    downloads: Path,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    wheels: tuple[release.Wheel, ...] = release.NVIDIA_WHEELS,
    allow_http: bool = False,
    need_free_gb: float = release.NVIDIA_DISK_GB,
) -> None:
    """Download, verify and unpack every wheel, then move the set into place. Raises NvidiaPackError or
    ``fetch.CancelledError``; partly downloaded wheels are kept so the next try continues them."""
    nvidia.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(nvidia).free / 1e9
    if free < need_free_gb:
        raise NvidiaPackError(f"not enough free disk space: about {need_free_gb:.0f} GB needed, {free:.1f} GB free")
    for old in nvidia.glob("site.new-*"):
        shutil.rmtree(old, ignore_errors=True)
    stage = nvidia / f"site.new-{os.getpid()}-{int(time.time())}"
    stage.mkdir()
    total = sum(w.size for w in wheels) if all(w.size for w in wheels) else 0
    done = 0
    try:
        for wheel in wheels:
            base = done

            def step(got: int, _all: int, base: int = base, name: str = wheel.name) -> None:
                if progress:
                    progress(name, base + got, total)

            archive = downloads / wheel.url.rsplit("/", 1)[-1]
            try:
                fetch.download(wheel.url, archive, wheel.sha256, wheel.size, step, cancel, allow_http, MAX_WHEEL_UNPACKED)

                def choose(name: str, keep: tuple[str, ...] = wheel.keep) -> str | None:
                    return name if name.startswith(keep) else None

                unpack.extract(archive, stage, choose, MAX_WHEEL_UNPACKED, cancel=cancel)
            except fetch.CancelledError:
                raise
            except InterruptedError as exc:
                raise fetch.CancelledError("cancelled") from exc
            except (fetch.DownloadError, unpack.UnsafeArchiveError, OSError) as exc:
                raise NvidiaPackError(f"{wheel.name}: {exc}") from exc
            done += wheel.size or archive.stat().st_size
            with contextlib.suppress(OSError):
                archive.unlink()  # unpacked: the download is not needed again
        if not (stage / "onnxruntime" / "__init__.py").is_file():
            raise NvidiaPackError("the downloaded set holds no ONNX Runtime")
        (stage / READY).write_text(json.dumps({"wheels": _pins(wheels)}, indent=1), encoding="utf-8")
        site = site_dir(nvidia)
        try:
            if site.exists():
                old = nvidia / f"site.old-{os.getpid()}"
                os.replace(site, old)
                shutil.rmtree(old, ignore_errors=True)
            os.replace(stage, site)
        except OSError as exc:
            raise NvidiaPackError(f"the libraries could not be put in place ({exc}); close the stream and try again") from exc
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def remove(nvidia: Path) -> None:
    shutil.rmtree(nvidia, ignore_errors=True)
