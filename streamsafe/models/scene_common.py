"""SigLIP2 scene model (shared by every backend): prompts, image preparation, probabilities.

SigLIP2 zero-shot scene model, one forward pass per call:
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
from PIL import Image

SIZE = 224  # siglip2-base-patch16-224 processor: PIL bilinear resize to 224x224, x/255, (x - 0.5) / 0.5

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

PROMPTS = SWIM_PROMPTS + EXPLICIT_PROMPTS + OTHER_PROMPTS
SWIM = slice(0, len(SWIM_PROMPTS))
EXPLICIT = slice(len(SWIM_PROMPTS), len(SWIM_PROMPTS) + len(EXPLICIT_PROMPTS))


def prepare(rgb: np.ndarray) -> np.ndarray:
    """RGB frame -> (1, 3, 224, 224) float32 as the Hugging Face SigLIP processor makes it."""
    im = np.asarray(Image.fromarray(rgb).resize((SIZE, SIZE), Image.BILINEAR), np.float32)
    return np.ascontiguousarray(((im / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None], np.float32)


def probs(embedding: np.ndarray, text: np.ndarray, scale: float, bias: float) -> tuple[np.ndarray, np.ndarray]:
    """(normalised image embedding, softmax over every prompt) from a raw pooled embedding (d,)."""
    v = embedding.astype(np.float32) / max(float(np.linalg.norm(embedding)), 1e-12)
    logits = text @ v * scale + bias
    e = np.exp(logits - logits.max())
    return v, e / e.sum()
