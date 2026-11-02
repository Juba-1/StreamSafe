"""SigLIP2 zero-shot scene prior: probability that the frame is a beach / pool scene."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from streamsafe.models.torch_backend.device import DEVICE, HALF

SWIM_PROMPTS = [
    "a video game screenshot of a sunny beach with people",
    "a video game screenshot of a swimming pool",
    "a video game screenshot of people in swimsuits at the beach",
    "a video game screenshot of the sea shore",
]
OTHER_PROMPTS = [
    "a video game screenshot of a city street",
    "a video game screenshot of a dark room indoors",
    "a video game screenshot of a jungle camp",
    "a video game screenshot of an office or shop",
    "a video game screenshot of a first person view of hands and a weapon",
]


class SigLIPScene:
    def __init__(self, repo: str = "google/siglip2-base-patch16-224") -> None:
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(repo, torch_dtype=HALF).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(repo)
        text = self.processor(text=SWIM_PROMPTS + OTHER_PROMPTS, padding="max_length", max_length=64, return_tensors="pt")
        with torch.inference_mode():
            t = self.model.get_text_features(**{k: v.to(DEVICE) for k, v in text.items()})
            t = getattr(t, "pooler_output", t)  # newer transformers return a model output
            self.text = torch.nn.functional.normalize(t.float(), dim=-1)

    def pswim(self, rgb: np.ndarray) -> float:
        pixels = self.processor(images=[Image.fromarray(rgb)], return_tensors="pt")["pixel_values"].to(DEVICE, HALF)
        with torch.inference_mode():
            v = self.model.vision_model(pixel_values=pixels).pooler_output.float()
            v = torch.nn.functional.normalize(v, dim=-1)
            logits = v @ self.text.T * self.model.logit_scale.exp().float() + self.model.logit_bias.float()
            return float(torch.softmax(logits, -1)[0, : len(SWIM_PROMPTS)].sum())
