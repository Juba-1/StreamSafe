"""ViTPose-base on ONNX Runtime (pre/post-processing: ``vitpose_common``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from streamsafe.models import vitpose_common as vp
from streamsafe.models.base import Keypoints
from streamsafe.models.onnx_backend.session import Shapes, Trt, open_session
from streamsafe.types import Box

MAX_BATCH = 16  # people per run: a fixed upper end for the TensorRT engine (and for GPU memory)
INPUT = "pixels"  # the input's name in tools/export_onnx.py
# the pose model's batch range for TensorRT (the engine and ``streamsafe.prepare`` must ask for the same one)
SHAPES: Shapes = {INPUT: tuple((n, 3, vp.INPUT_H, vp.INPUT_W) for n in (1, 2, MAX_BATCH))}


class OnnxViTPose:
    def __init__(self, path: str | Path, allow_cpu: bool = False, trt: Trt | None = None) -> None:
        self.session = open_session(path, allow_cpu, trt, SHAPES)
        self.input_name = self.session.get_inputs()[0].name
        if self.input_name != INPUT:
            raise ValueError(f"{Path(path).name}: input is {self.input_name!r}, expected {INPUT!r}: re-export")

    def __call__(self, rgb: np.ndarray, boxes: list[Box]) -> list[Keypoints]:
        if not boxes:
            return []
        x, geometry = vp.prepare(rgb, boxes)
        out = [self.session.run(None, {self.input_name: x[i : i + MAX_BATCH]})[0] for i in range(0, len(x), MAX_BATCH)]
        return vp.keypoints(np.concatenate(out).astype(np.float32), geometry)
