"""What the engine needs from each neural model. Any backend (PyTorch, ONNX Runtime, test stubs)
that implements these protocols can be plugged in; nothing else in the engine imports a framework."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from streamsafe.types import Box

Keypoints = tuple[np.ndarray, np.ndarray]  # (17, 2) float32 COCO keypoints, (17,) float32 scores


class PersonDetector(Protocol):
    def __call__(self, rgb: np.ndarray) -> list[tuple[Box, np.ndarray]]:
        """People in an RGB analysis frame: (box, bool mask of the frame size) each."""
        ...


class RiskEye(Protocol):
    thr: float  # operating threshold the heads were calibrated for

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """RGB person crops -> L2-normalised embeddings (n, d)."""
        ...

    def prisk(self, emb: np.ndarray) -> np.ndarray:
        """Embeddings (n, d) -> probability that each person must be covered (n,)."""
        ...


class PoseEstimator(Protocol):
    def __call__(self, rgb: np.ndarray, boxes: list[Box]) -> list[Keypoints]:
        """COCO-17 keypoints for every box, in frame coordinates."""
        ...


class SceneClassifier(Protocol):
    def __call__(self, rgb: np.ndarray) -> tuple[float, float | None]:
        """(probability of a beach / pool scene, probability of an explicit scene or None when the model has
        no trained explicit-scene head) for an RGB frame."""
        ...


class HudReader(Protocol):
    def __call__(self, bgr: np.ndarray) -> str:
        """Text in the game's prompt / subtitle areas of a BGR frame (any size), '' when none."""
        ...
