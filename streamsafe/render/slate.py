"""Full-screen fail-closed frame ("back in a moment")."""

from __future__ import annotations

import cv2
import numpy as np

BACKGROUND = (40, 28, 22)  # BGR
TEXT = "StreamSafe - back in a moment"


def make_slate(width: int, height: int) -> np.ndarray:
    slate = np.full((height, width, 3), BACKGROUND, np.uint8)
    scale = 1.6 * width / 1920
    (tw, th), _ = cv2.getTextSize(TEXT, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    origin = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(slate, TEXT, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), max(1, round(3 * width / 1920)))
    return slate
