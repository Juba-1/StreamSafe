"""Eye input preparation shared by every backend."""

from __future__ import annotations

import numpy as np
from PIL import Image


def prepare(crops: list[np.ndarray], width: int, height: int) -> np.ndarray:
    """RGB person crops -> (n, 3, height, width) float32 in [0, 1].
    PIL bicubic (antialiased) exactly as in head training - cv2.resize changes the scores."""
    batch = np.stack([np.asarray(Image.fromarray(c).resize((width, height), Image.BICUBIC)) for c in crops])
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2), np.float32) / np.float32(255.0)
