"""Scene-cut detector: colour-histogram distance plus mean pixel change on a thumbnail."""

from __future__ import annotations

import cv2
import numpy as np

HIST_CUT = 0.5  # Bhattacharyya distance that counts as a cut when the picture also changed...
PIXEL_CUT = 35.0  # ...by this many grey levels on average
HIST_ALWAYS = 0.85  # a cut regardless of pixel change


class CutDetector:
    def __init__(self) -> None:
        self._hist: np.ndarray | None = None
        self._gray: np.ndarray | None = None

    def __call__(self, frame: np.ndarray) -> bool:
        """True when ``frame`` (BGR, any size) starts a new shot."""
        small = cv2.resize(frame, (160, 90))
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        gray = cv2.cvtColor(cv2.resize(small, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        cut = False
        if self._hist is not None:
            dist = cv2.compareHist(self._hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            cut = (dist > HIST_CUT and float(np.abs(gray - self._gray).mean()) > PIXEL_CUT) or dist > HIST_ALWAYS
        self._hist, self._gray = hist, gray
        return cut
