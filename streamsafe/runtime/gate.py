"""Fail-closed output gate: turns (frame, decisions) into the frame viewers see.

  * a frame is drawn only with decisions from its own shot, at most ``max_gap`` frames away, and only when the
    decisions around it are at most ``max_span`` frames apart (a too-slow analyzer fails closed);
  * no usable decision, or a failed one -> the last protected frame is repeated for up to ``hold``
    frames (a one-frame flash between two cuts), then the full-screen slate;
  * a usable decision with ``scene`` (explicit scene) -> the full-screen scene card; frames up to
    ``scene_back_s`` BEFORE that decision are covered too (they are still waiting in the delay);
  * ``muted`` (game audio): on for every full-screen frame, ``mute_lead_s`` before a scene cover starts
    and ``mute_tail_s`` after it ends;
  * everything here is deterministic code - models only feed the decisions.
Thread-safe: in realtime mode decisions, cut flags and renders come from different threads.
"""

from __future__ import annotations

import os
import threading
import time
from bisect import bisect_right, insort
from collections import defaultdict

import cv2
import numpy as np

from streamsafe.config import BrandConfig, RenderConfig, SceneConfig
from streamsafe.render.compositor import DrawLog, compose, usable_decisions
from streamsafe.render.lookahead import backfill
from streamsafe.render.slate import brand_frames
from streamsafe.render.smooth import TrackSmoother
from streamsafe.runtime.control import Live
from streamsafe.types import ANALYSIS_SIZE, Decision

FULL_HOLD_S = 1.5  # whole-screen mode: the cover stays this long after the last person who needed covering

SCENE_LOOK = 150  # frames: how far ahead a scene decision is searched (late evidence reaches back this far at most)


class OutputGate:
    def __init__(
        self,
        cfg: RenderConfig,
        output_size: tuple[int, int],
        keep_log: bool = False,
        scene: SceneConfig | None = None,
        fps: float = 30.0,
        brand: BrandConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.size = output_size
        self.shown_boxes: list[tuple[float, float, float, float]] = []
        self.zones: Live = Live()  # the streamer's webcam zones: no person cover there (set while live)
        self.scale = output_size[0] / ANALYSIS_SIZE[0]
        self.slate, self.card = brand_frames(*output_size, brand or BrandConfig())
        self.smoother = TrackSmoother(cfg.smooth_alpha, cfg.smooth_beta, cfg.smooth_reset, cfg.smooth_fast) if cfg.smooth else None
        scene = scene or SceneConfig()
        self._back = round(scene.scene_back_s * fps)
        self._lead = round(scene.mute_lead_s * fps)
        self._tail = round(scene.mute_tail_s * fps)
        # the streamer's choices: COVER_MODE=screen -> anyone who needs covering covers the WHOLE screen (held
        # FULL_HOLD_S after the last one); manual -> the panic key: the whole screen, from the next frame shown
        self.full_mode = os.environ.get("COVER_MODE", "") == "screen"
        self._full_hold = round(FULL_HOLD_S * fps)
        self._full_until = -1
        self.manual = False
        self.muted = False  # game audio state for the frame rendered last
        self._last_full = -(10**9)  # last frame shown full-screen (slate or scene card)
        self._mute_from = -1  # first frame of the current mute interval
        self.stats: defaultdict[str, list] = defaultdict(list)
        self.log: DrawLog | None = [] if keep_log else None
        self._lock = threading.Lock()
        self._decisions: dict[int, Decision] = {}
        self._keys: list[int] = []  # sorted frame ids of the stored decisions
        self._cuts: dict[int, bool] = {}
        self._last: np.ndarray | None = None
        self._held = 0
        self._shown = -1  # last frame id rendered

    @property
    def n_decisions(self) -> int:
        with self._lock:
            return len(self._keys)

    def mark_cut(self, frame_id: int, cut: bool) -> None:
        with self._lock:
            self._cuts[frame_id] = cut

    def add_decision(self, decision: Decision) -> None:
        if (self.zones.zones or self.zones.allow) and decision.layers:  # the streamer's own picture (webcam) is never dressed
            w, h = self.size
            decision.layers = [g for g in decision.layers if not self.zones.in_zone(g.centroid[1] / w, g.centroid[0] / h)]
        with self._lock:
            if decision.frame_id not in self._decisions:
                insort(self._keys, decision.frame_id)
            self._decisions[decision.frame_id] = decision
            if self.cfg.lookahead and decision.ok and decision.layers:
                self._look_back(decision)

    def _look_back(self, new: Decision) -> None:
        """Copy the new decision's garments into earlier decisions of the same shot that can still
        protect a frame not shown yet (see render/lookahead.py)."""
        f = new.frame_id
        earlier = [
            self._decisions[k]
            for k in self._keys
            if f - self.cfg.lookahead_frames <= k < f and k + self.cfg.max_gap > self._shown and not self._cut_between(k, f)
        ]
        for d in backfill(new, earlier, self.scale):
            self._decisions[d.frame_id] = d
            self.stats["backfilled"].append(d.backfilled)

    def forget_before(self, frame_id: int) -> None:
        """Drop decisions and cut flags no future frame can use."""
        with self._lock:
            while self._keys and self._keys[0] < frame_id:
                del self._decisions[self._keys.pop(0)]
            for k in [k for k in self._cuts if k < frame_id]:
                del self._cuts[k]

    def _cut_between(self, a: int, b: int) -> bool:
        """A scene cut on any frame in (a, b]."""
        return any(self._cuts.get(i, False) for i in range(a + 1, b + 1))

    def render(self, v: int, frame: np.ndarray) -> np.ndarray:
        """Protected output for frame ``v`` (``frame`` is drawn on in place and may be returned)."""
        t0 = time.perf_counter()
        with self._lock:
            self._shown = v
            ids = usable_decisions(v, self._keys, self._cut_between, self.cfg.max_gap, self.cfg.max_span)
            decisions = [self._decisions[k] for k in ids]
            scene = any(d.scene for d in decisions) or self._scene_ahead(v, 0)
            scene_soon = self._scene_ahead(v, self._lead)
        if self.full_mode and any(d.layers for d in decisions):
            self._full_until = v + self._full_hold
        if self.manual or v <= self._full_until:
            scene = True
        if not decisions or not all(d.ok for d in decisions):
            if self._last is not None and self._held < self.cfg.hold:
                out = self._last
                self._held += 1
                self.stats["held_frames"].append(v)
            else:
                out = self.slate
                self.stats["failclosed_frames"].append(v)
        elif scene:
            self._held = 0
            out = self.card
            self.stats["scene_frames"].append(v)
        else:
            self._held = 0
            gray = None
            if self.cfg.motion:  # same analysis-size gray the analyzer used for its references
                gray = cv2.cvtColor(cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if self.smoother is not None:
                if self._cuts.get(v, False):
                    self.smoother.clear()
                self.smoother.prune(v)
            out = compose(frame, v, decisions, gray, self.cfg.mc_min, self.scale, self.log, self.smoother)
            h, w = frame.shape[:2]
            self.shown_boxes = [  # where the person covers of this frame are (fractions), for the window
                (g.x / w, g.y / h, (g.x + g.mask.shape[1]) / w, (g.y + g.mask.shape[0]) / h) for g in decisions[-1].layers
            ]
        if out is self.slate or out is self.card:
            self.shown_boxes = []
        self._last = out
        if out is self.slate or out is self.card:
            self._last_full = v
        self._set_mute(v, v - self._last_full <= self._tail or scene_soon)
        self.stats["render_ms"].append(1000 * (time.perf_counter() - t0))
        return out

    def _scene_ahead(self, v: int, margin: int) -> bool:
        """A later scene decision whose cover reaches back to frame v (- margin): it starts ``scene_back_s``
        before its own frame, or before its evidence frame (``scene_from``, a late HUD read). Caller holds
        the lock."""
        i = bisect_right(self._keys, v)
        while i < len(self._keys) and self._keys[i] <= v + SCENE_LOOK:
            d = self._decisions[self._keys[i]]
            if d.scene and (d.frame_id if d.scene_from is None else min(d.scene_from, d.frame_id)) - self._back - margin <= v:
                return True
            i += 1
        return False

    def _set_mute(self, v: int, muted: bool) -> None:
        if muted and not self.muted:
            self._mute_from = v
        elif self.muted and not muted:
            self.stats["mute_intervals"].append([self._mute_from, v - 1])
        if muted:
            self.stats["muted_frames"].append(v)
        self.muted = muted

    def finish(self) -> None:
        """Closes an open mute interval (call once after the last frame)."""
        if self.muted:
            self.stats["mute_intervals"].append([self._mute_from, self._shown])
            self.muted = False
