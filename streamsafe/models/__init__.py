"""Neural models behind small protocols (see ``base``). ``build_models`` loads the default
PyTorch backend lazily, so the rest of the engine (and the tests) run without torch."""

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
    from streamsafe.models.torch_backend import load_torch_models

    return load_torch_models(cfg)
