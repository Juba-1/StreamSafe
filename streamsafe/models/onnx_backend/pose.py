"""ViTPose-base on ONNX Runtime (pre/post-processing: ``vitpose_common``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from streamsafe.models import vitpose_common as vp
from streamsafe.models.base import Keypoints
from streamsafe.models.onnx_backend.session import open_session
from streamsafe.types import Box


class OnnxViTPose:
    def __init__(self, path: str | Path, allow_cpu: bool = False) -> None:
        self.session = open_session(path, allow_cpu)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray, boxes: list[Box]) -> list[Keypoints]:
        if not boxes:
            return []
        x, geometry = vp.prepare(rgb, boxes)
        heatmaps = self.session.run(None, {self.input_name: x})[0].astype(np.float32)
        return vp.keypoints(heatmaps, geometry)
