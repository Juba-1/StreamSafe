"""SigLIP2 scene model on ONNX Runtime: the vision tower runs per call; the prompt embeddings were computed once
at export (``scene_text.npy``). What it computes and why: ``streamsafe/models/scene_common.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from streamsafe.models import scene_common as sc
from streamsafe.models.heads import LinearHead
from streamsafe.models.onnx_backend.session import open_session


class OnnxSigLIPScene:
    def __init__(self, folder: str | Path, head: LinearHead | None = None, allow_cpu: bool = False) -> None:
        folder = Path(folder)
        self.session = open_session(folder / "scene_vision.onnx", allow_cpu)
        self.input_name = self.session.get_inputs()[0].name
        self.text = np.load(folder / "scene_text.npy").astype(np.float32)
        if self.text.shape[0] != len(sc.PROMPTS):
            raise ValueError(f"scene_text.npy has {self.text.shape[0]} prompts, the engine has {len(sc.PROMPTS)}: re-export")
        self.scale, self.bias = (float(v) for v in np.load(folder / "scene_logit.npy"))
        self.head = head

    def embed_probs(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        emb = self.session.run(None, {self.input_name: sc.prepare(rgb)})[0][0]
        return sc.probs(emb, self.text, self.scale, self.bias)

    def __call__(self, rgb: np.ndarray) -> tuple[float, float | None]:
        emb, p = self.embed_probs(rgb)
        explicit = float(self.head(emb)) if self.head is not None else None
        return float(p[sc.SWIM].sum()), explicit
