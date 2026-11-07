"""PyTorch implementations of the model protocols (the configuration measured in runs 20-27)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streamsafe.config import Config
    from streamsafe.models import Models


def load_torch_models(cfg: Config) -> Models:
    from streamsafe.models import Models
    from streamsafe.models.torch_backend.detector import RFDETRDetector
    from streamsafe.models.torch_backend.eye import RadioEye
    from streamsafe.models.torch_backend.pose import ViTPose
    from streamsafe.models.torch_backend.scene import SigLIPScene

    return Models(
        detector=RFDETRDetector(cfg.detector.det_model, cfg.detector.det_thr),
        eye=RadioEye(cfg.eye.eye_repo, cfg.eye.eye_w, cfg.eye.eye_h, cfg.eye.head_dir),
        pose=ViTPose(),
        scene=SigLIPScene(head=_scene_head(cfg)) if cfg.eye.scene_prior or cfg.scene.scene_guard else None,
        hud=_hud(cfg),
    )


def _scene_head(cfg: Config):  # a trained explicit-scene head, when the head folder has one
    from pathlib import Path

    from streamsafe.models.heads import SCENE_FILE, LinearHead

    path = Path(cfg.eye.head_dir) / SCENE_FILE
    return LinearHead.load(path) if cfg.eye.head_dir and path.exists() else None


def _hud(cfg: Config):  # optional dependency: imported only when enabled
    if not cfg.scene.hud_ocr:
        return None
    from streamsafe.models.ocr_hud import RapidHud  # fails loudly when enabled but not installed

    return RapidHud()
