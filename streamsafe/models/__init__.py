"""Neural models behind small protocols (see ``base``). ``build_models`` loads the backend chosen in the config
(``BACKEND``: "torch" or "onnx") lazily, so the rest of the engine (and the tests) run without either."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from streamsafe.models.base import HudReader, PersonDetector, PoseEstimator, RiskEye, SceneClassifier

if TYPE_CHECKING:
    from streamsafe.config import Config

__all__ = ["HudReader", "Models", "PersonDetector", "PoseEstimator", "RiskEye", "SceneClassifier", "build_models"]


@dataclass
class Models:
    detector: PersonDetector
    eye: RiskEye
    pose: PoseEstimator
    scene: SceneClassifier | None = None
    hud: HudReader | None = None


def build_models(cfg: Config) -> Models:
    if cfg.backend.backend == "onnx":
        from streamsafe.models.onnx_backend import load_onnx_models

        return load_onnx_models(cfg)
    if cfg.backend.backend == "torch":
        from streamsafe.models.torch_backend import load_torch_models

        return load_torch_models(cfg)
    raise ValueError(f"unknown BACKEND {cfg.backend.backend!r} (torch / onnx)")


def scene_head(cfg: Config):
    """The trained explicit-scene head, when the head folder has one (any backend)."""
    from pathlib import Path

    from streamsafe.models.heads import SCENE_FILE, LinearHead

    path = Path(cfg.eye.head_dir) / SCENE_FILE
    return LinearHead.load(path) if cfg.eye.head_dir and path.exists() else None


def hud_reader(cfg: Config):
    """Optional HUD reader: imported only when enabled, fails loudly when enabled but not installed."""
    if not cfg.scene.hud_ocr:
        return None
    from streamsafe.models.ocr_hud import RapidHud

    return RapidHud()
