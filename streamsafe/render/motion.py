"""Moves each garment to where its person is in the frame being shown.

At decision time we keep the person's own pixels (background flattened to its mean, so the static
background does not dominate); at render time phase correlation of that reference against the
same window of the shown frame gives the person's shift. Untrustworthy estimates return None and
the compositor falls back to interpolation between decisions, or leaves the garment in place.
"""

from __future__ import annotations

import cv2
import numpy as np

from streamsafe.types import ANALYSIS_SIZE, Box, Layer, MotionRef

REF_MAX = 128  # reference side length (px)
REF_MIN = 16
MIN_MASK = 50
MIN_CONTRAST = 4.0  # grey levels: below this the person has no texture and no outline to follow
MAX_SHIFT = 0.3  # of the window size


def make_motion_ref(gray: np.ndarray, mask: np.ndarray, box: Box) -> MotionRef | None:
    """Reference from a grayscale analysis frame, or None when motion cannot be measured."""
    aw, ah = ANALYSIS_SIZE
    x1, y1, x2, y2 = box
    ex = int(0.3 * max(x2 - x1, 0.5 * (y2 - y1))) + 4
    rx1, ry1, rx2, ry2 = max(0, x1 - ex), max(0, y1 - ex), min(aw, x2 + ex), min(ah, y2 + ex)
    g = gray[ry1:ry2, rx1:rx2].astype(np.float32)
    m = mask[ry1:ry2, rx1:rx2]
    if m.sum() < MIN_MASK or g.size == 0:
        return None
    fill = float(g[~m].mean()) if (~m).any() else float(g.mean())
    g = np.where(m, g, fill)  # background -> its mean: the person's outline stays as contrast
    if np.abs(g[m] - fill).mean() < MIN_CONTRAST and g[m].std() < MIN_CONTRAST:
        return None
    k = REF_MAX / max(g.shape)
    size = (max(REF_MIN, int(g.shape[1] * k)) & ~1, max(REF_MIN, int(g.shape[0] * k)) & ~1)
    return MotionRef(
        region=(rx1, ry1, rx2, ry2),
        ref=cv2.resize(g, size, interpolation=cv2.INTER_AREA),
        size=size,
        window=cv2.createHanningWindow(size, cv2.CV_32F),
    )


def motion_shift(layer: Layer, gray: np.ndarray | None, min_response: float, scale: float) -> tuple[float, float] | None:
    """(dy, dx) in output pixels that moves ``layer`` onto its person in ``gray`` (analysis-size frame)."""
    ref = layer.motion
    if ref is None or gray is None:
        return None
    x1, y1, x2, y2 = ref.region
    cur = cv2.resize(gray[y1:y2, x1:x2], ref.size, interpolation=cv2.INTER_AREA).astype(np.float32)
    (sx, sy), response = cv2.phaseCorrelate(ref.ref, cur, ref.window)
    if response < min_response:
        return None
    dx, dy = sx * (x2 - x1) / ref.size[0], sy * (y2 - y1) / ref.size[1]
    if abs(dx) > MAX_SHIFT * (x2 - x1) or abs(dy) > MAX_SHIFT * (y2 - y1):
        return None
    return scale * dy, scale * dx
