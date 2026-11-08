"""C-RADIOv4-H embedding + gated linear risk heads.

bf16 weights with autocast on the GPU (pure bf16 without autocast fails in RADIO's patch
embedder); falls back to fp32 automatically when bf16 does not run on the card.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from streamsafe.models.heads import RiskHeads
from streamsafe.models.torch_backend.device import DEVICE


class RadioEye:
    def __init__(self, repo: str, width: int, height: int, head_dir: str, revision: str) -> None:
        from transformers import AutoModel

        self.width, self.height = width, height
        self.model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True).to(DEVICE).eval()
        self.bf16 = False
        if DEVICE == "cuda":
            self.model = self.model.to(torch.bfloat16)
            self.bf16 = True
            try:
                self._forward(torch.rand(2, 3, height, width, device=DEVICE))
            except Exception:
                self.model = self.model.float()
                self.bf16 = False
        self.heads = RiskHeads.load(head_dir)
        self.thr = self.heads.thr

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            return self.model(x)[0].float()

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        # PIL bicubic (antialiased) exactly as in head training - cv2.resize changes the scores
        size = (self.width, self.height)
        batch = np.stack([np.asarray(Image.fromarray(c).resize(size, Image.BICUBIC)) for c in crops])
        x = torch.from_numpy(batch).to(DEVICE).permute(0, 3, 1, 2).float().div(255)
        return torch.nn.functional.normalize(self._forward(x), dim=-1).cpu().numpy()

    def prisk(self, emb: np.ndarray) -> np.ndarray:
        return self.heads.prisk(emb)
