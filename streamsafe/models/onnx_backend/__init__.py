"""ONNX Runtime implementations of the model protocols: the same models as the PyTorch backend, exported with
``tools/export_onnx.py``, running on NVIDIA (CUDA, optionally TensorRT), any DirectX 12 GPU on Windows (DirectML)
or the CPU.

Before anything is loaded, every file is checked against ``manifest.json`` (sha256): a damaged or swapped model
file stops the engine at start (fail closed), it never runs with unknown weights.

Hashing gigabytes at every start cost 10-20 s (load bench 25 Nov, runs/load_bench_r2): ``sha256_known`` hashes a
file once and keeps the result with the file's size and modification time (``STAMPS`` in the same folder); a later
start that finds the same size and time takes the kept digest. Any change to the file (an update, a damaged
download, a swap) changes its time or size, and it is hashed again.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streamsafe.config import Config
    from streamsafe.models import Models
    from streamsafe.models.onnx_backend.session import Shapes, Trt

REQUIRED = ("detector.onnx", "eye.onnx", "eye.onnx.data", "pose.onnx", "scene_vision.onnx", "scene_text.npy", "scene_logit.npy")


def verify(folder: Path, pins: dict[str, str]) -> dict:
    """Check every file listed in manifest.json (size + sha256) and that the eye / scene / pose pins match the
    engine's config. Raises ValueError on any mismatch."""
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{manifest_path} missing: re-export the models")
    files = json.loads(manifest_path.read_text())["files"]
    missing = [name for name in REQUIRED if name not in files]
    if missing:
        raise ValueError(f"model folder {folder} lacks {missing}")
    for name, meta in files.items():
        path = folder / name
        if not path.exists() or path.stat().st_size != meta["bytes"] or sha256_known(path) != meta["sha256"]:
            raise ValueError(f"model file {path} is missing or damaged (size / sha256 mismatch)")
        repo = meta.get("repo")
        if repo in pins and meta.get("revision") != pins[repo]:
            raise ValueError(f"{name} was exported from {repo}@{meta.get('revision')}, the engine is pinned to {pins[repo]}")
    return files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


STAMPS = ".sha256_stamps.json"


def sha256_known(path: Path) -> str:
    """sha256 of ``path``, hashed only when its size or modification time differs from the last time."""
    st = path.stat()
    stamp = [st.st_size, st.st_mtime_ns]
    book = path.parent / STAMPS
    try:
        known = json.loads(book.read_text())
    except (OSError, ValueError):
        known = {}
    entry = known.get(path.name) if isinstance(known, dict) else None
    if isinstance(entry, list) and len(entry) == 3 and entry[:2] == stamp:
        return str(entry[2])
    digest = sha256(path)
    if not isinstance(known, dict):
        known = {}
    known[path.name] = [*stamp, digest]
    tmp = book.with_name(book.name + ".tmp")
    try:  # a folder that cannot be written only means hashing again next time
        tmp.write_text(json.dumps(known))
        tmp.replace(book)
    except OSError:
        pass
    return digest


def checked_files(cfg: Config) -> dict:
    """The model folder verified against its manifest and against what the config asks for."""
    folder = Path(cfg.backend.onnx_dir)
    pins = {
        cfg.eye.eye_repo: cfg.eye.eye_revision,
        cfg.scene.scene_repo: cfg.scene.scene_revision,
        cfg.garment.pose_repo: cfg.garment.pose_revision,
    }
    files = verify(folder, pins)
    if cfg.backend.eye_fp16 and "eye16.onnx" not in files:
        raise ValueError("EYE_FP16=1 but the model folder has no eye16.onnx: export it (tools/export_onnx.py ... eye16)")
    want = f"rfdetr RFDETRSeg{cfg.detector.det_model}"
    if files["detector.onnx"].get("model") != want:
        raise ValueError(f"detector.onnx is {files['detector.onnx'].get('model')!r}, the config wants {want!r}")
    return files


def eye_file(cfg: Config) -> str:
    return "eye16.onnx" if cfg.backend.eye_fp16 else "eye.onnx"


def uses_scene(cfg: Config) -> bool:
    return cfg.eye.scene_prior or cfg.scene.scene_guard


def trt_settings(cfg: Config, files: dict, **changes) -> Trt:
    """The TensorRT settings of this config (``changes``: what the prepare step sets differently)."""
    from streamsafe.models.onnx_backend.session import Trt

    b = cfg.backend
    cache = b.trt_cache or str(Path(b.onnx_dir) / "trt_cache")
    return Trt(**{**dict(cache=cache, fp16=b.trt_fp16, files=files, build=b.trt_build, libs=b.trt_libs), **changes})


def trt_models(cfg: Config) -> list[tuple[str, str, Shapes | None]]:
    """(label, file, batch range) of every model that gets a TensorRT engine with this config, in loading order.
    The engine below and ``streamsafe.prepare`` both go by this list, so what was prepared is what is loaded."""
    from streamsafe.models.onnx_backend import eye, pose

    models: list[tuple[str, str, Shapes | None]] = [("scene", "scene_vision.onnx", None)] if uses_scene(cfg) else []
    return [
        *models,
        ("detector", "detector.onnx", None),  # fixed input shapes: no batch range
        ("eye", eye_file(cfg), eye.shapes(cfg.eye.eye_w, cfg.eye.eye_h)),
        ("pose", "pose.onnx", pose.SHAPES),
    ]


def load_onnx_models(cfg: Config) -> Models:
    from streamsafe.models import Models, hud_reader, scene_head
    from streamsafe.models.onnx_backend import session
    from streamsafe.models.onnx_backend.detector import OnnxRFDETR
    from streamsafe.models.onnx_backend.eye import OnnxRadioEye
    from streamsafe.models.onnx_backend.pose import OnnxViTPose
    from streamsafe.models.onnx_backend.scene import OnnxSigLIPScene

    folder = Path(cfg.backend.onnx_dir)
    files = checked_files(cfg)
    cpu = cfg.backend.allow_cpu
    trt = trt_settings(cfg, files) if cfg.backend.trt else None
    session.reset()
    scene = OnnxSigLIPScene(folder, head=scene_head(cfg), allow_cpu=cpu, trt=trt) if uses_scene(cfg) else None
    models = Models(
        detector=OnnxRFDETR(folder / "detector.onnx", cfg.detector.det_thr, cpu, trt),
        eye=OnnxRadioEye(folder / eye_file(cfg), cfg.eye.eye_w, cfg.eye.eye_h, cfg.eye.head_dir, cpu, trt),
        pose=OnnxViTPose(folder / "pose.onnx", cpu, trt),
        scene=scene,
        hud=hud_reader(cfg),
    )
    for note in session.notes():  # a model off its fast path, or an engine built at this start: said at once,
        print(f"STREAMSAFE_DEVICE {session.device_summary()} | {note}", file=sys.stderr, flush=True)  # not only in the final report
    return models
