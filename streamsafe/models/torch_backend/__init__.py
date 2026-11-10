"""PyTorch implementations of the model protocols (the configuration measured in runs 20-27)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streamsafe.config import Config
    from streamsafe.models import Models


def load_torch_models(cfg: Config) -> Models:
    from streamsafe.models import Models, hud_reader, scene_head
    from streamsafe.models.torch_backend.detector import RFDETRDetector
    from streamsafe.models.torch_backend.eye import RadioEye
    from streamsafe.models.torch_backend.pose import ViTPose
    from streamsafe.models.torch_backend.scene import SigLIPScene

    scene = None  # one model serves the beach prior and the scene layer
    if cfg.eye.scene_prior or cfg.scene.scene_guard:
        scene = SigLIPScene(cfg.scene.scene_repo, cfg.scene.scene_revision, head=scene_head(cfg))
    return Models(
        detector=RFDETRDetector(cfg.detector.det_model, cfg.detector.det_thr),
        eye=RadioEye(cfg.eye.eye_repo, cfg.eye.eye_w, cfg.eye.eye_h, cfg.eye.head_dir, cfg.eye.eye_revision),
        pose=ViTPose(cfg.garment.pose_repo, cfg.garment.pose_revision),
        scene=scene,
        hud=hud_reader(cfg),
    )
