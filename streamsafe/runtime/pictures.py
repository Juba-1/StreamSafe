"""Pictures the streamer always wants covered (a game's map, a phone screen in the game, a menu): each one is a
small saved picture at a fixed place on the screen. Every shown frame is compared with it there, BEFORE the
person covers are drawn; when it matches, that place (or the whole screen) is covered opaque for as long as it
shows, plus ``HOLD_S`` so it never flickers.

One picture costs a crop, a resize to at most ``SIDE`` px and one correlation: well under a millisecond.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

SIDE = 96  # templates are compared at this width at most (fast, and blind to small video noise)
MATCH = 0.80  # correlation of a picture with detail
MATCH_FLAT = 0.93  # 1 - mean difference, for a nearly plain picture (correlation means nothing there)
HOLD_S = 0.6
FULL = (0.0, 0.0, 1.0, 1.0)


def _gray(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (3, 3), 0)


def load_template(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    h, w = img.shape
    if w > SIDE:
        w, h = SIDE, max(4, round(h * SIDE / w))
    return _gray(img, (max(4, w), max(4, h)))


def score(region: np.ndarray, tpl: np.ndarray) -> tuple[float, bool]:
    """(similarity, flat): ``region`` already at the template's size and gray."""
    if float(tpl.std()) < 4.0:
        return 1.0 - float(np.mean(cv2.absdiff(region, tpl))) / 255.0, True
    return float(cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)[0, 0]), False


class PictureWatch:
    """``pictures``: (x0, y0, x1, y1, file, full) from the live control file. Call once per shown frame."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._templates: dict[str, np.ndarray | None] = {}
        self._until: dict[str, float] = {}
        self.matches = 0  # frames where a picture was found (session report)

    def __call__(self, frame: np.ndarray, pictures: tuple) -> list[tuple[float, float, float, float]]:
        if not pictures:
            return []
        h, w = frame.shape[:2]
        now = self._clock()
        out = []
        for x0, y0, x1, y1, file, full in pictures:
            if file not in self._templates and Path(file).is_file():
                self._templates[file] = load_template(file)
            tpl = self._templates.get(file)
            if tpl is None:
                continue
            a, b, c, d = round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)
            if c - a < 2 or d - b < 2:
                continue
            region = _gray(frame[b:d, a:c], (tpl.shape[1], tpl.shape[0]))
            s, flat = score(region, tpl)
            if s >= (MATCH_FLAT if flat else MATCH):
                self._until[file] = now + HOLD_S
                self.matches += 1
            if self._until.get(file, 0.0) > now:
                out.append(FULL if full else (x0, y0, x1, y1))
        return out
