"""SigLIP2 zero-shot scene model, one forward pass per call:
  * beach / pool probability (the eye's threshold prior),
  * explicit-scene probability (the scene layer's whole-screen cover, ``analysis/scene.py``).
Probabilities = softmax over all prompts, summed per group. The OTHER group holds look-alikes that must
NOT count as explicit (night clubs, bars, beaches are handled per person by the garments).
The explicit probability is reported ONLY with a trained head (``scene_h.txt`` next to the eye heads: a
linear head on the same normalised image embedding, trained on labeled game frames). The zero-shot explicit
score is kept for bench tools but never drives the cover: on GTA V clips it scored clothed cutscene arguments
0.9-0.98 and a helicopter over sandy hills 0.67 (run 42 sheets), which would blank normal gameplay."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from streamsafe.models.heads import LinearHead
from streamsafe.models.torch_backend.device import DEVICE, HALF

SWIM_PROMPTS = [
    "a video game screenshot of a sunny beach with people",
    "a video game screenshot of a swimming pool",
    "a video game screenshot of people in swimsuits at the beach",
    "a video game screenshot of the sea shore",
]
EXPLICIT_PROMPTS = [
    "a video game screenshot of a stripper dancing on a pole",
    "a video game screenshot of a lap dance in a strip club",
    "a video game screenshot of a topless woman",
    "a video game screenshot of a sex scene",
    "a video game screenshot of a close-up of a woman's body in lingerie",
]
OTHER_PROMPTS = [
    "a video game screenshot of a city street",
    "a video game screenshot of a dark room indoors",
    "a video game screenshot of a jungle camp",
    "a video game screenshot of an office or shop",
    "a video game screenshot of a first person view of hands and a weapon",
    "a video game screenshot of a crowded nightclub dance floor",
    "a video game screenshot of people talking in a bar",
    "a video game screenshot of a car driving",
    "a video game screenshot of a close-up of a face",
]


class SigLIPScene:
    def __init__(self, repo: str = "google/siglip2-base-patch16-224", head: LinearHead | None = None) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(repo, torch_dtype=HALF).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(repo)
        prompts = SWIM_PROMPTS + EXPLICIT_PROMPTS + OTHER_PROMPTS
        text = self.processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt")
        with torch.inference_mode():
            t = self.model.get_text_features(**{k: v.to(DEVICE) for k, v in text.items()})
            t = getattr(t, "pooler_output", t)  # newer transformers return a model output
            self.text = torch.nn.functional.normalize(t.float(), dim=-1)
        n_s, n_e = len(SWIM_PROMPTS), len(EXPLICIT_PROMPTS)
        self._swim, self._explicit = slice(0, n_s), slice(n_s, n_s + n_e)
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
