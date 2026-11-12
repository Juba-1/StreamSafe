"""Build or verify the TensorRT engines before a stream needs them (NVIDIA edition).

    python -m streamsafe.prepare --models DIR [--cache DIR] [--check] [--json]

TensorRT compiles each model into an engine for this graphics card, driver and TensorRT version: about three
minutes for all of them on an RTX 3090 (run 57). Run by the installer and by the app before every start, this
step pays those minutes where a progress bar can be shown, never at "go live". When every engine is already
there and fits this PC it only re-hashes the files (seconds) and builds nothing.

It prints one line per event, flushed, for the app to show:

    preparing 2/4: eye...
    ready 2/4: eye (engine built in 58 s (not built yet))
    PREPARE_OK 4/4 engines ready

and on failure a last line with the reason and what to do:

    PREPARE_FAILED eye: the engine could not be built (...)

``--json`` prints the same events as one JSON object per line (``event``, ``step``, ``of``, ``name``, ``detail``).
``--check`` builds nothing and needs no GPU work: it only says whether the engines are ready.

Exit code: 0 = every engine ready; 2 = failed (the engine would run that model on CUDA and say so in its report);
3 = ``--check`` only: not ready, run without ``--check``.

The engines are the ones the engine itself will ask for: same model list, same settings (``EYE_FP16``,
``TRT_FP16``, ``TRT_CACHE``, ``TRT_LIBS`` from the environment), same key (``models/onnx_backend/engines.py``).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from streamsafe.config import Config

READY, FAILED, NOT_READY = 0, 2, 3
# estimate, to be replaced by the sizes a GPU run reports: the engines of all models plus one build in progress
NEED_FREE_GB = 4.0

Emit = Callable[..., None]  # emit(event, step=0, of=0, name="", detail="")


def status(cfg: Config, quick: bool = False) -> list[tuple[str, str, str]]:
    """(label, model file, why its engine is not ready or "") for every model that gets an engine. Reads the
    disk, the driver version and the library versions only; no model is loaded.
    Raises ``NotUsableError`` when TensorRT cannot be used on this PC at all, ValueError for a bad model folder."""
    from streamsafe.models.onnx_backend import checked_files

    files = _manifest(Path(cfg.backend.onnx_dir)) if quick else checked_files(cfg)
    return [(label, file, why) for label, file, _, why in _engines(cfg, files, quick)]


def _engines(cfg: Config, files: dict, quick: bool = False) -> list[tuple[str, str, Path, str]]:
    """(label, model file, engine folder, why not ready or "") per model, by the key the engine itself will use."""
    from streamsafe.models.onnx_backend import engines, trt_models, trt_settings
    from streamsafe.models.onnx_backend.session import trt_options

    folder = Path(cfg.backend.onnx_dir)
    trt = trt_settings(cfg, files)
    ident = engines.identity(trt.libs)
    rows = []
    for label, file, shapes in trt_models(cfg):
        key = engines.engine_key(file, engines.model_shas(folder / file, files), trt_options(trt, shapes), ident)
        rows.append((label, file, engines.folder(trt.cache, file, key), engines.why_not_ready(trt.cache, file, key, quick)))
    return rows


def _manifest(folder: Path) -> dict:
    """The manifest as written, for the quick check (the full check hashes every model file against it)."""
    try:
        return json.loads((folder / "manifest.json").read_text())["files"]
    except (OSError, KeyError) as exc:
        raise ValueError(f"{folder / 'manifest.json'} missing or unreadable: reinstall the models") from exc


def prepare(cfg: Config, emit: Emit, check_only: bool = False) -> int:
    """Make every engine ready, one model at a time. Returns the exit code."""
    from streamsafe.models.onnx_backend import checked_files, engines, session, trt_models, trt_settings

    folder = Path(cfg.backend.onnx_dir)
    emit("verifying", detail="model files")
    try:
        files = checked_files(cfg)
        rows = _engines(cfg, files)
    except (ValueError, OSError, engines.NotUsableError) as exc:
        emit("failed", name="tensorrt" if isinstance(exc, engines.NotUsableError) else "models", detail=str(exc))
        return FAILED
    waiting = [label for label, _, _, why in rows if why]
    if check_only:
        for i, (label, _, _, why) in enumerate(rows, 1):
            emit("not_ready" if why else "ready", i, len(rows), label, why or "verified")
        emit("todo" if waiting else "done", of=len(rows), detail=", ".join(waiting))
        return NOT_READY if waiting else READY
    # strict: a model that does not end up on TensorRT is an error here, never a quiet CUDA session
    trt = trt_settings(cfg, files, build=True, strict=True)
    if waiting:
        problem = _cannot_build(trt.cache, emit)
        if problem:
            emit("failed", name="tensorrt", detail=problem)
            return FAILED
    shapes = {label: s for label, _, s in trt_models(cfg)}
    for i, (label, file, _, why) in enumerate(rows, 1):
        if not why:
            emit("ready", i, len(rows), label, "verified")
            continue
        emit("preparing", i, len(rows), label, why)
        try:
            # one model at a time and freed again: a build needs GPU memory that the other models would hold
            model = session.open_session(folder / file, False, trt, shapes[label])
            note = session.OPENED[-1].note
            del model
            gc.collect()
        except Exception as exc:
            emit("failed", i, len(rows), label, " ".join(str(exc).split())[:400])
            return FAILED
        emit("ready", i, len(rows), label, note)

    try:  # the proof is on disk, not in this process: every engine must now pass the check the engine will make
        rows = _engines(cfg, files)
        left = [f"{label} ({why})" for label, _, _, why in rows if why]
    except (OSError, engines.NotUsableError) as exc:
        left = [str(exc)]
    if left:
        emit("failed", name="cache", detail="built but not stored: " + "; ".join(left))
        return FAILED
    removed = engines.sweep(trt.cache, [path for _, _, path, _ in rows])
    emit("done", of=len(rows), detail=f"removed {len(removed)} old engine folder(s)" if removed else "")
    return READY


def _cannot_build(cache: str, emit: Emit) -> str:
    """Why no engine can be built here at all ("" = go ahead): said before minutes are spent on a build."""
    from streamsafe.models.onnx_backend.session import PROVIDERS, providers

    try:
        if providers()[0] != PROVIDERS["cuda"]:
            return "TensorRT needs an NVIDIA graphics card and the NVIDIA runtime (onnxruntime-gpu)"
    except (ImportError, RuntimeError) as exc:
        return f"ONNX Runtime cannot be used ({exc})"
    try:
        os.makedirs(cache, exist_ok=True)
        probe = Path(cache) / f".write-test-{os.getpid()}"
        probe.write_bytes(b"x")
        probe.unlink()
        free = shutil.disk_usage(cache).free / 1e9
    except OSError as exc:
        return f"the engine folder {cache} cannot be written ({exc}): set TRT_CACHE to a folder of your own user"
    if free < NEED_FREE_GB:  # a warning only: the figure is an estimate, and a real shortage fails the build with its reason
        emit("warning", detail=f"{free:.1f} GB free at {cache}; the engines may need about {NEED_FREE_GB:.0f} GB")
    return ""


def _printer(as_json: bool) -> Emit:
    def emit(event: str, step: int = 0, of: int = 0, name: str = "", detail: str = "") -> None:
        if as_json:
            line = json.dumps(dict(event=event, step=step, of=of, name=name, detail=detail), ensure_ascii=False)
        elif event == "done":
            line = f"PREPARE_OK {of}/{of} engines ready" + (f" ({detail})" if detail else "")
        elif event == "todo":
            line = f"PREPARE_NEEDED not ready: {detail}"
        elif event == "failed":
            line = f"PREPARE_FAILED {name}: {detail}"
        elif event == "preparing":
            line = f"preparing {step}/{of}: {name}..." + (f" ({detail})" if detail else "")
        elif step:
            line = f"{event.replace('_', ' ')} {step}/{of}: {name}" + (f" ({detail})" if detail else "")
        else:
            line = f"{event}: {detail}"
        print(line, flush=True)

    return emit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="streamsafe.prepare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.environ.get("ONNX_DIR", ""), help="ONNX model folder (default: ONNX_DIR)")
    ap.add_argument("--cache", help="engine folder (default: TRT_CACHE, or <models>/trt_cache)")
    ap.add_argument("--check", action="store_true", help="only report whether the engines are ready; build nothing")
    ap.add_argument("--json", action="store_true", help="one JSON object per line, for the installer / app")
    args = ap.parse_args(argv)
    emit = _printer(args.json)
    if not args.models:
        emit("failed", name="models", detail="no model folder: pass --models DIR or set ONNX_DIR")
        return FAILED
    overrides = dict(onnx_dir=args.models, trt=True, **({"trt_cache": args.cache} if args.cache else {}))
    return prepare(Config.from_env(**overrides), emit, args.check)


if __name__ == "__main__":
    sys.exit(main())
