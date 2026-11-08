"""Steadies where each person's garment is drawn, frame to frame (no image work: a few numbers per person).

The measured placement (motion-compensated, or interpolated between decisions) is noisy: phase correlation
wobbles by a pixel or two, and when the garment switches to the next decision's shape its centre can step.
An alpha-beta (constant-velocity) filter per person removes the wobble without trailing a person who moves
steadily, and a switch to a new shape is eased in over a few frames instead of popping. The filter only
smooths small corrections: the farther a measurement is from the prediction (a sudden sprint or turn), the
more of it can be taken at once (``fast_frac``, off by default: run 49). A real jump (cut,
re-detection, a measurement far from the prediction) resets the filter to the measurement at once, so the
garment never drifts away from its person.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _Track:
    v: int  # frame of the last placement
    kf: int  # decision whose shape was drawn
    cy: float
    cx: float
    vy: float = 0.0
    vx: float = 0.0


class TrackSmoother:
    def __init__(self, alpha: float, beta: float, reset_frac: float, fast_frac: float = 0.0, forget: int = 30) -> None:
        self.alpha, self.beta, self.reset_frac, self.fast_frac, self.forget = alpha, beta, reset_frac, fast_frac, forget
        self._tracks: dict[int, _Track] = {}

    def clear(self) -> None:
        self._tracks.clear()

    def place(self, tid: int, v: int, kf: int, cy: float, cx: float, size: float) -> tuple[float, float]:
        """Steadied centre for the measured centre (cy, cx) of person ``tid`` in frame ``v`` (output px)."""
        t = self._tracks.get(tid)
        if t is None or v - t.v != 1:  # new person, or frames skipped: start from the measurement
            self._tracks[tid] = _Track(v, kf, cy, cx)
            return cy, cx
        py, px = t.cy + t.vy, t.cx + t.vx
        ry, rx = cy - py, cx - px
        if math.hypot(ry, rx) > self.reset_frac * max(1.0, size):  # not the same smooth motion: follow at once
            self._tracks[tid] = _Track(v, kf, cy, cx)
            return cy, cx
        a = self.alpha
        if self.fast_frac > 0:  # small residual = wobble -> alpha; fast_frac x size or more = real motion -> taken whole
            a = max(a, min(1.0, math.hypot(ry, rx) / (self.fast_frac * max(1.0, size))))
        t.cy, t.cx = py + a * ry, px + a * rx
        if kf == t.kf:  # a new shape's centre step is eased in but is not a velocity
            t.vy, t.vx = t.vy + self.beta * ry, t.vx + self.beta * rx
        t.v, t.kf = v, kf
        return t.cy, t.cx

    def prune(self, v: int) -> None:
        for tid in [k for k, t in self._tracks.items() if v - t.v > self.forget]:
            del self._tracks[tid]
