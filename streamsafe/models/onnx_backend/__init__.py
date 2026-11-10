"""ONNX Runtime implementations of the model protocols: the same models as the PyTorch backend, exported with
``tools/export_onnx.py``, running on NVIDIA (CUDA), any DirectX 12 GPU on Windows (DirectML) or the CPU.

Before anything is loaded, every file is checked against ``manifest.json`` (sha256): a damaged or swapped model
file stops the engine at start (fail closed), it never runs with unknown weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streamsafe.config import Config
    from streamsafe.models import Models

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
        if not path.exists() or path.stat().st_size != meta["bytes"] or sha256(path) != meta["sha256"]:
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


def load_onnx_models(cfg: Config) -> Models:
    from streamsafe.models import Models, hud_reader, scene_head
    from streamsafe.models.onnx_backend.detector import OnnxRFDETR
    from streamsafe.models.onnx_backend.eye import OnnxRadioEye
    from streamsafe.models.onnx_backend.pose import OnnxViTPose
    from streamsafe.models.onnx_backend.scene import OnnxSigLIPScene

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
    cpu = cfg.backend.allow_cpu
    eye_file = "eye16.onnx" if cfg.backend.eye_fp16 else "eye.onnx"
    scene = None
    if cfg.eye.scene_prior or cfg.scene.scene_guard:
        scene = OnnxSigLIPScene(folder, head=scene_head(cfg), allow_cpu=cpu)
    return Models(
        detector=OnnxRFDETR(folder / "detector.onnx", cfg.detector.det_thr, cpu),
        eye=OnnxRadioEye(folder / eye_file, cfg.eye.eye_w, cfg.eye.eye_h, cfg.eye.head_dir, cpu),
        pose=OnnxViTPose(folder / "pose.onnx", cpu),
        scene=scene,
        hud=hud_reader(cfg),
    )
