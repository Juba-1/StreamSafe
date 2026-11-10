"""C-RADIOv4-H on ONNX Runtime + the same gated risk heads as the PyTorch backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from streamsafe.models import eye_common
from streamsafe.models.heads import RiskHeads
from streamsafe.models.onnx_backend.session import open_session

MAX_BATCH = 16  # crops per run: bounds GPU memory on small cards


class OnnxRadioEye:
    def __init__(self, path: str | Path, width: int, height: int, head_dir: str, allow_cpu: bool = False) -> None:
        self.session = open_session(path, allow_cpu)
        self.input_name = self.session.get_inputs()[0].name
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
