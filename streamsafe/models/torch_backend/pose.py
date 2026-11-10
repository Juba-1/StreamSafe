"""ViTPose-base on PyTorch, every person of a frame in one batch (pre/post-processing: ``vitpose_common``)."""

from __future__ import annotations

import numpy as np
import torch

from streamsafe.models import vitpose_common as vp
from streamsafe.models.base import Keypoints
from streamsafe.models.torch_backend.device import DEVICE, HALF
from streamsafe.types import Box


class ViTPose:
    def __init__(self, repo: str, revision: str) -> None:
        from transformers import VitPoseForPoseEstimation

        self.model = VitPoseForPoseEstimation.from_pretrained(repo, revision=revision).to(DEVICE, HALF).eval()

    def __call__(self, rgb: np.ndarray, boxes: list[Box]) -> list[Keypoints]:
        if not boxes:
            return []
        x, geometry = vp.prepare(rgb, boxes)
        with torch.inference_mode():
            heatmaps = self.model(pixel_values=torch.from_numpy(x).to(DEVICE, HALF)).heatmaps.float().cpu().numpy()
        return vp.keypoints(heatmaps, geometry)
