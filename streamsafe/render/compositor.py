"""Draws the garments of the decisions around a frame onto that frame.

``usable_decisions``: frame v is protected by the closest decision at or before it and the closest
after it, each only if it is within ``max_gap`` frames AND no scene cut lies between them.
``compose``: per person, the nearer decision's garment is moved onto the person by motion
compensation; when that fails the garment is interpolated between the two decisions (same track
in both) or left in place. The farther copy is drawn only when the nearer one had no reliable
motion estimate - drawing both would show a misplaced double ("ghost").
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from streamsafe.render.motion import motion_shift
from streamsafe.render.smooth import TrackSmoother
from streamsafe.types import Decision, Layer

MAX_INTERP_JUMP = 0.6  # of the layer size: larger centroid jumps are not the same motion

DrawLog = list[list[int]]


def usable_decisions(
    v: int, keys: Sequence[int], cut_between: Callable[[int, int], bool], max_gap: int, max_span: int | None = None
) -> list[int]:
    """Frame ids of the decisions (k0 <= v < k1, sorted ``keys``) allowed to protect frame v.
    ``max_span``: two decisions of the same shot farther apart than this protect nothing in between (the
    analyzer is too slow: a person walking in between them would show) -> the caller fails closed."""
    i = bisect_right(keys, v)
    k0 = keys[i - 1] if i > 0 else None
    k1 = keys[i] if i < len(keys) else None
    if max_span is not None:
        if k0 is not None and k1 is not None and k0 < v and k1 - k0 > max_span and not cut_between(k0, k1):
            return []
        max_gap = min(max_gap, max_span)  # one-sided (next to a cut, or the analyzer far behind): same limit
    use = []
    if k0 is not None and v - k0 <= max_gap and not cut_between(k0, v):
        use.append(k0)
    if k1 is not None and k1 - v <= max_gap and not cut_between(v, k1):
        use.append(k1)
    return use


def draw(out: np.ndarray, layer: Layer, dy: float = 0.0, dx: float = 0.0) -> None:
    """Opaque copy of the layer's cloth into ``out`` (in place), clipped to the frame."""
    oh, ow = out.shape[:2]
    y0, x0 = round(layer.y + dy), round(layer.x + dx)
    h, w = layer.shape
    a, b, c, d = max(0, y0), max(0, x0), min(oh, y0 + h), min(ow, x0 + w)
    if c <= a or d <= b:
        return
    sy, sx = a - y0, b - x0
    cv2.copyTo(layer.fabric[sy : sy + c - a, sx : sx + d - b], layer.mask_u8[sy : sy + c - a, sx : sx + d - b], out[a:c, b:d])


def _interpolation_offsets(v: int, decisions: list[Decision]) -> dict[int, tuple[float, float]]:
    """Per layer (by id()), the offset that places it at its interpolated position in frame v."""
    offsets: dict[int, tuple[float, float]] = {}
    if len(decisions) != 2:
        return offsets
    d0, d1 = decisions
    t = (v - d0.frame_id) / max(1, d1.frame_id - d0.frame_id)
    later = {layer.tid: layer for layer in d1.layers}
    for a in d0.layers:
        b = later.get(a.tid)
        if b is None:
            continue
        dy, dx = b.centroid[0] - a.centroid[0], b.centroid[1] - a.centroid[1]
        if abs(dy) + abs(dx) <= MAX_INTERP_JUMP * max(a.shape):
            offsets[id(a)] = (t * dy, t * dx)
            offsets[id(b)] = (-(1 - t) * dy, -(1 - t) * dx)
    return offsets


def compose(
    frame: np.ndarray,
    v: int,
    decisions: list[Decision],
    gray: np.ndarray | None,
    mc_min: float,
    scale: float,
    log: DrawLog | None = None,
    smoother: TrackSmoother | None = None,
) -> np.ndarray:
    """Draws onto ``frame`` (in place) and returns it. ``decisions`` in time order (1 or 2);
    ``gray`` = analysis-size grayscale of ``frame`` (None disables motion compensation)."""
    offsets = _interpolation_offsets(v, decisions)
    if len(decisions) == 2 and decisions[1].frame_id - v < v - decisions[0].frame_id:
        decisions = [decisions[1], decisions[0]]  # nearer decision first
    tracked: set[int] = set()  # people already drawn with a reliable motion estimate
    placed: set[int] = set()  # people already drawn at least once
    for dec in decisions:
        for layer in dec.layers:
            if layer.tid in tracked:
                continue
            shift = motion_shift(layer, gray, mc_min, scale)
            if shift is not None:
                tracked.add(layer.tid)
            dy, dx = shift if shift is not None else offsets.get(id(layer), (0.0, 0.0))
            if smoother is not None and layer.tid not in placed:  # the person's main copy only
                cy, cx = layer.centroid[0] + dy, layer.centroid[1] + dx
                sy, sx = smoother.place(layer.tid, v, dec.frame_id, cy, cx, max(layer.shape))
                dy, dx = dy + sy - cy, dx + sx - cx
            placed.add(layer.tid)
            draw(frame, layer, dy, dx)
            if log is not None:
                h, w = layer.shape
                log.append(
                    [
                        v,
                        int(layer.tid),
                        int(dec.frame_id),
                        round((layer.y + dy) / scale),
                        round((layer.x + dx) / scale),
                        int(h / scale),
                        int(w / scale),
                        int(shift is not None),
                        0,
                    ]
                )
    return frame
