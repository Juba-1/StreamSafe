"""Fail-closed output gate: turns (frame, decisions) into the frame viewers see.

  * a frame is drawn only with decisions from its own shot, at most ``max_gap`` frames away;
  * no usable decision, or a failed one -> the last protected frame is repeated for up to ``hold``
    frames (a one-frame flash between two cuts), then the full-screen slate;
  * everything here is deterministic code - models only feed the decisions.
Thread-safe: in realtime mode decisions, cut flags and renders come from different threads.
"""

from __future__ import annotations

import threading
import time
from bisect import insort
from collections import defaultdict

import cv2
import numpy as np

from streamsafe.config import RenderConfig
from streamsafe.render.compositor import DrawLog, compose, usable_decisions
from streamsafe.render.lookahead import backfill
from streamsafe.render.slate import make_slate
from streamsafe.types import ANALYSIS_SIZE, Decision


class OutputGate:
    def __init__(self, cfg: RenderConfig, output_size: tuple[int, int], keep_log: bool = False) -> None:
        self.cfg = cfg
        self.scale = output_size[0] / ANALYSIS_SIZE[0]
        self.slate = make_slate(*output_size)
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
            ids = usable_decisions(v, self._keys, self._cut_between, self.cfg.max_gap)
            decisions = [self._decisions[k] for k in ids]
        if not decisions or not all(d.ok for d in decisions):
            if self._last is not None and self._held < self.cfg.hold:
                out = self._last
                self._held += 1
                self.stats["held_frames"].append(v)
            else:
                out = self.slate
                self.stats["failclosed_frames"].append(v)
        else:
            self._held = 0
            gray = None
            if self.cfg.motion:  # same analysis-size gray the analyzer used for its references
                gray = cv2.cvtColor(cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            out = compose(frame, v, decisions, gray, self.cfg.mc_min, self.scale, self.log)
        self._last = out
        self.stats["render_ms"].append(1000 * (time.perf_counter() - t0))
        return out
