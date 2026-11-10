"""C-RADIOv4-H on ONNX Runtime + the same gated risk heads as the PyTorch backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from streamsafe.models import eye_common
from streamsafe.models.heads import RiskHeads
from streamsafe.models.onnx_backend.session import Trt, open_session

MAX_BATCH = 16  # crops per run: bounds GPU memory on small cards
INPUT = "pixels"  # the input's name in tools/export_onnx.py (TensorRT wants the batch range before the model is open)


class OnnxRadioEye:
    def __init__(
        self, path: str | Path, width: int, height: int, head_dir: str, allow_cpu: bool = False, trt: Trt | None = None, usual: int = 4
    ) -> None:
        """``usual``: the batch size TensorRT tunes its engine for (every size up to MAX_BATCH still runs)."""
        batches = (1, max(1, min(usual, MAX_BATCH)), MAX_BATCH)
        self.session = open_session(path, allow_cpu, trt, {INPUT: tuple((n, 3, height, width) for n in batches)})
        self.input_name = self.session.get_inputs()[0].name
        if self.input_name != INPUT:
            raise ValueError(f"{Path(path).name}: input is {self.input_name!r}, expected {INPUT!r}: re-export")
        self.width, self.height = width, height
        self.heads = RiskHeads.load(head_dir)
        self.thr = self.heads.thr

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        x = eye_common.prepare(crops, self.width, self.height)
        out = [self.session.run(None, {self.input_name: x[i : i + MAX_BATCH]})[0] for i in range(0, len(x), MAX_BATCH)]
        emb = np.concatenate(out).astype(np.float32)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    def prisk(self, emb: np.ndarray) -> np.ndarray:
        return self.heads.prisk(emb)
