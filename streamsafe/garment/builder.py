"""Builds one ready-to-draw layer per covered person, once per decision.

Two looks, chosen per person with hysteresis (``choose_mode``):
  * "small" - far / small person: opaque plain cloth over the whole (slightly dilated) body;
  * "full"  - pose-shaped dress (robe when the pose is unreliable) with cloth shading.
Continuity between decisions: the skeleton is smoothed, and a garment never shrinks in a single
step (it is united with the previous decision's own garment, moved to the new box).
Safety contract: the layer always contains the person's own mask pixels in the covered band; it
only gives up pixels belonging to a clearly safe person standing in front / merged into the box.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from streamsafe.analysis.tracks import GarmentMemory
from streamsafe.config import GarmentConfig
from streamsafe.garment.fabric import fabric, load_logo, parse_color, put_logo
from streamsafe.garment.shapes import cover_shape, robe_region
from streamsafe.models.base import Keypoints
from streamsafe.types import ANALYSIS_SIZE, Layer, Person

HARD_RISK = 0.85  # a hard-risky person never gives pixels away to neighbours
SMALL_HYSTERESIS = (0.8, 1.25)  # small->full needs height >= small_h*1.25; full->small needs < small_h*0.8
SHAPE_MEMORY_SCALE = (0.8, 1.25)  # previous garment reused only when the person's size barely changed
MIN_PIXELS = 30
EDGE_SHADE = 0.82
K11, K5, K3 = np.ones((11, 11), np.uint8), np.ones((5, 5), np.uint8), np.ones((3, 3), np.uint8)


class SafePerson(NamedTuple):
    mask: np.ndarray
    box: tuple[int, int, int, int]
    area: int


def choose_mode(height: int, memory: GarmentMemory, small_h: int) -> str:
    lo, hi = SMALL_HYSTERESIS
    limit = small_h * (hi if memory.mode == "small" else lo if memory.mode == "full" else 1.0)
    memory.mode = "small" if height < limit else "full"
    return memory.mode


class GarmentBuilder:
    def __init__(self, cfg: GarmentConfig, output_scale: float) -> None:
        self.cfg = cfg
        self.scale = output_scale  # output px per analysis px
        self.color = parse_color(cfg.color)
        self.logo = load_logo(cfg.logo)

    def build(
        self,
        im: np.ndarray,
        person: Person,
        mode: str,
        memory: GarmentMemory,
        safe: list[SafePerson],
        decision: int,
        cut: bool,
        pose: Keypoints | None = None,
    ) -> Layer | None:
        """Layer for one covered person (``im`` = BGR analysis frame), or None if nothing to cover."""
        aw, ah = ANALYSIS_SIZE
        x1, y1, x2, y2 = person.box
        pad = max(12, int(0.35 * (y2 - y1)))
        wy1, wy2, wx1, wx2 = max(0, y1 - pad), min(ah, y2 + pad), max(0, x1 - pad), min(aw, x2 + pad)
        own = person.mask[wy1:wy2, wx1:wx2]

        safe_px = np.zeros(own.shape, bool)
        merged = np.zeros(own.shape, bool)  # safe people mostly inside this person's mask
        for s in safe:
            sx1, sy1, sx2, sy2 = s.box
            if sx2 < wx1 or sx1 > wx2 or sy2 < wy1 or sy1 > wy2:
                continue
            local = s.mask[wy1:wy2, wx1:wx2]
            safe_px |= local
            if (local & own).sum() >= 0.5 * max(1, s.area):
                merged |= local

        body = cv2.dilate(own.astype(np.uint8), K5 if mode == "small" else K11) > 0
        if person.p < HARD_RISK:
            body = (body & ~safe_px) | (own & ~merged)  # own pixels go only to a safe person merged into the box
        safe_near = cv2.dilate(safe_px.astype(np.uint8), K5).astype(bool) if safe_px.any() else safe_px

        if mode == "small":
            cover = body & ~(safe_near & ~own)
            if cover.sum() < MIN_PIXELS:
                return None
            return self._plain_layer(cover, (wy1, wx1), person.tid)

        kp, sc = self._smoothed_pose(pose, memory, decision, cut)
        kp_local = kp - np.array([wx1, wy1], np.float32) if kp is not None else None
        try:
            cover, keep = cover_shape(body, kp_local, sc)
        except Exception:  # never fail the decision because of geometry: cover the body
            cover, keep = body, np.zeros_like(body)
        cover = cover & ~(safe_near & ~body) & ~keep  # never spills onto a clearly safe person
        cover = self._with_shape_memory(cover, body, own, safe_near, person, memory, (wy1, wx1), decision, cut)
        if cover.sum() < MIN_PIXELS:
            return None
        return self._dress_layer(im[wy1:wy2, wx1:wx2], cover, (wy1, wx1), person.tid)

    # -- continuity ----------------------------------------------------------------------------
    def _smoothed_pose(
        self, pose: Keypoints | None, memory: GarmentMemory, decision: int, cut: bool
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        kp, sc = pose if pose is not None else (None, None)
        if kp is not None and self.cfg.kp_smooth > 0:
            prev = memory.keypoints
            if prev is not None and memory.keypoints_at == decision - 1 and not cut:
                ok = (sc > 0.3) & (prev[1] > 0.3)
                kp = kp.copy()
                kp[ok] = (1 - self.cfg.kp_smooth) * kp[ok] + self.cfg.kp_smooth * prev[0][ok]
            memory.keypoints, memory.keypoints_at = (kp, sc), decision
        return kp, sc

    def _with_shape_memory(
        self,
        cover: np.ndarray,
        body: np.ndarray,
        own: np.ndarray,
        safe_near: np.ndarray,
        person: Person,
        memory: GarmentMemory,
        origin: tuple[int, int],
        decision: int,
        cut: bool,
    ) -> np.ndarray:
        """cover U (last decision's own garment moved / scaled to this box, kept where the body is)."""
        fresh = cover.copy()
        x1, y1, x2, y2 = person.box
        wy1, wx1 = origin
        if self.cfg.shape_memory and memory.cover is not None and not cut and memory.cover_at == decision - 1:
            prev, pbox, (py1, px1) = memory.cover
            s = (y2 - y1) / max(1, pbox[3] - pbox[1])
            if SHAPE_MEMORY_SCALE[0] <= s <= SHAPE_MEMORY_SCALE[1]:
                cy0, cx0 = (pbox[1] + pbox[3]) / 2 - py1, (pbox[0] + pbox[2]) / 2 - px1
                cy1, cx1 = (y1 + y2) / 2 - wy1, (x1 + x2) / 2 - wx1
                m = np.float32([[s, 0, cx1 - s * cx0], [0, s, cy1 - s * cy0]])
                moved = cv2.warpAffine(prev.astype(np.uint8), m, (cover.shape[1], cover.shape[0]), flags=cv2.INTER_NEAREST)
                cover = cover | (moved.astype(bool) & body & ~(safe_near & ~own))
        memory.cover, memory.cover_at = (fresh, person.box, origin), decision  # one step only: never accumulates
        return cover

    # -- layers --------------------------------------------------------------------------------
    def _plain_layer(self, cover: np.ndarray, origin: tuple[int, int], tid: int) -> Layer:
        ys, xs = np.nonzero(cover)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = cover[y1:y2, x1:x2]
        cloth = np.empty((*sub.shape, 3), np.uint8)
        cloth[:] = np.clip(np.array(self.color, np.float32) * 0.9, 0, 255).astype(np.uint8)
        return self._layer(sub, cloth, (y1 + origin[0], x1 + origin[1]), tid)

    def _dress_layer(self, im: np.ndarray, cover: np.ndarray, origin: tuple[int, int], tid: int) -> Layer:
        """Shaded cloth over ``cover``; shading at 1/fabric_ds resolution, outline at full resolution."""
        ys, xs = np.nonzero(cover)
        h, w = cover.shape
        y1, y2, x1, x2 = max(0, ys.min() - 4), min(h, ys.max() + 5), max(0, xs.min() - 4), min(w, xs.max() + 5)
        sub, crop = cover[y1:y2, x1:x2], im[y1:y2, x1:x2]
        out_size = self._out_size(sub)
        try:
            cloth = self._shade(crop, sub, tid)
            cloth = cv2.resize(cloth, out_size, interpolation=cv2.INTER_LINEAR)
        except Exception:  # shading is cosmetic: a flat opaque cloth is always a valid cover
            cloth = np.empty((out_size[1], out_size[0], 3), np.uint8)
            cloth[:] = self.color
        edge = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_GRADIENT, K3).astype(bool) & sub
        return self._layer(sub, cloth, (y1 + origin[0], x1 + origin[1]), tid, edge)

    def _shade(self, crop: np.ndarray, sub: np.ndarray, seed: int) -> np.ndarray:
        ds = self.cfg.fabric_ds if sub.shape[0] >= 80 else 1
        crop_d, sub_d = crop, sub
        if ds > 1:
            size = (max(8, sub.shape[1] // ds), max(8, sub.shape[0] // ds))
            sub_d = cv2.resize(sub.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) > 127
            if sub_d.sum() >= 60:
                crop_d = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            else:
                sub_d = sub
        _, info = robe_region(sub_d)
        cloth = fabric(crop_d, sub_d, info, self.color, seed)
        if self.logo is not None:
            cloth = put_logo(cloth, sub_d, self.logo)
        return np.clip(cloth, 0, 255).astype(np.uint8)

    def _out_size(self, sub: np.ndarray) -> tuple[int, int]:
        return round(sub.shape[1] * self.scale), round(sub.shape[0] * self.scale)

    def _layer(self, sub: np.ndarray, cloth: np.ndarray, origin: tuple[int, int], tid: int, edge: np.ndarray | None = None) -> Layer:
        size = self._out_size(sub)
        mask = cv2.resize(sub.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        if cloth.shape[:2] != mask.shape:
            cloth = cv2.resize(cloth, size, interpolation=cv2.INTER_LINEAR)
        if edge is not None:
            e = cv2.resize(edge.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
            cloth[e] = (cloth[e] * EDGE_SHADE).astype(np.uint8)
        y, x = round(self.scale * origin[0]), round(self.scale * origin[1])
        yy, xx = np.nonzero(mask)
        return Layer(tid=tid, y=y, x=x, mask=mask, fabric=cloth, centroid=(float(yy.mean()) + y, float(xx.mean()) + x))
