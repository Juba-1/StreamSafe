"""Look-ahead: cover a person from the moment they appeared, not from the moment the eye was sure.

Every frame waits ``delay`` seconds before viewers see it. When a person's track turns risky in a
new decision, the earlier decisions of the same shot that are still waiting may show that person
uncovered. ``backfill`` gives each of them a copy of the new garment, moved and scaled onto the
person's box in that earlier decision. Plain code, no model runs: it costs a few array copies.

Only earlier decisions that saw the same track id (the tracker's identity) are changed; nothing is
added across a scene cut or to frames already shown.
"""

from __future__ import annotations

from dataclasses import replace

import cv2

from streamsafe.types import Box, Decision, Layer

MAX_SCALE_JUMP = 2.0  # a box that grew or shrank more than this is not trusted as the same person


def moved_layer(layer: Layer, src: Box, dst: Box, scale: float) -> Layer | None:
    """``layer`` (drawn for a person at ``src``, analysis px) placed onto the same person at ``dst``."""
    sh, dh = src[3] - src[1], dst[3] - dst[1]
    if sh <= 0 or dh <= 0 or not 1 / MAX_SCALE_JUMP <= dh / sh <= MAX_SCALE_JUMP:
        return None
    k = dh / sh
    h, w = layer.shape
    nh, nw = max(1, round(h * k)), max(1, round(w * k))
    mask, fabric = layer.mask, layer.fabric
    if (nh, nw) != (h, w):
        mask = cv2.resize(mask.view("uint8"), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
        fabric = cv2.resize(fabric, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # keep the garment's position relative to the box: same offset from the box's top-left, scaled
    oy, ox = layer.y - src[1] * scale, layer.x - src[0] * scale
    y, x = round(dst[1] * scale + oy * k), round(dst[0] * scale + ox * k)
    cy, cx = layer.centroid
    centroid = (dst[1] * scale + (cy - src[1] * scale) * k, dst[0] * scale + (cx - src[0] * scale) * k)
    return Layer(tid=layer.tid, y=y, x=x, mask=mask, fabric=fabric, centroid=centroid, motion=None)


def backfill(new: Decision, earlier: list[Decision], scale: float) -> list[Decision]:
    """Earlier decisions (same shot, not shown yet) that need a copy of a garment from ``new``,
    returned as replacements; decisions that need nothing are not returned."""
    boxes_now = dict(new.people)
    fresh = [layer for layer in new.layers if layer.tid in boxes_now]
    if not fresh:
        return []
    out = []
    for d in earlier:
        if not d.ok:
            continue
        covered = {layer.tid for layer in d.layers}
        then = dict(d.people)
        added = [
            m
            for layer in fresh
            if layer.tid not in covered and layer.tid in then
            for m in [moved_layer(layer, boxes_now[layer.tid], then[layer.tid], scale)]
            if m is not None
        ]
        if added:
            out.append(replace(d, layers=[*d.layers, *added], backfilled=d.backfilled + len(added)))
    return out
