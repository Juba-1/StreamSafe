"""SigLIP2 scene model on PyTorch. What it computes and why: ``streamsafe/models/scene_common.py``."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from streamsafe.models.heads import LinearHead
from streamsafe.models.scene_common import EXPLICIT, PROMPTS, SWIM
from streamsafe.models.torch_backend.device import DEVICE, HALF


class SigLIPScene:
    def __init__(self, repo: str, revision: str, head: LinearHead | None = None) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(repo, revision=revision, torch_dtype=HALF).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(repo, revision=revision)
        text = self.processor(text=PROMPTS, padding="max_length", max_length=64, return_tensors="pt")
        with torch.inference_mode():
            t = self.model.get_text_features(**{k: v.to(DEVICE) for k, v in text.items()})
            t = getattr(t, "pooler_output", t)  # newer transformers return a model output
            self.text = torch.nn.functional.normalize(t.float(), dim=-1)
        self._swim, self._explicit = SWIM, EXPLICIT
        self.head = head

    def embed_probs(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(normalised image embedding, softmax over every prompt) - bench tools read both."""
        pixels = self.processor(images=[Image.fromarray(rgb)], return_tensors="pt")["pixel_values"].to(DEVICE, HALF)
        with torch.inference_mode():
            v = self.model.vision_model(pixel_values=pixels).pooler_output.float()
            v = torch.nn.functional.normalize(v, dim=-1)
            logits = v @ self.text.T * self.model.logit_scale.exp().float() + self.model.logit_bias.float()
            return v[0].cpu().numpy(), torch.softmax(logits, -1)[0].cpu().numpy()

    def __call__(self, rgb: np.ndarray) -> tuple[float, float | None]:
        emb, p = self.embed_probs(rgb)
        explicit = float(self.head(emb)) if self.head is not None else None
        return float(p[self._swim].sum()), explicit
