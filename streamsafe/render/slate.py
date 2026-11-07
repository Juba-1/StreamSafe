"""Full-screen frames: the fail-closed slate ("back in a moment") and the scene card (explicit scene hidden)."""

from __future__ import annotations

import cv2
import numpy as np

BACKGROUND = (40, 28, 22)  # BGR
SLATE_TEXT = "StreamSafe - back in a moment"
SCENE_BACKGROUND = (52, 36, 30)
SCENE_TEXT = "StreamSafe - scene hidden"


def make_slate(width: int, height: int, text: str = SLATE_TEXT, background: tuple[int, int, int] = BACKGROUND) -> np.ndarray:
    slate = np.full((height, width, 3), background, np.uint8)
    scale = 1.6 * width / 1920
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    origin = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(slate, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), max(1, round(3 * width / 1920)))
    return slate


def make_scene_card(width: int, height: int) -> np.ndarray:
    return make_slate(width, height, SCENE_TEXT, SCENE_BACKGROUND)
