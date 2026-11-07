"""The HUD reader on its own thread, so OCR (CPU, 0.1-1 s per read) never stalls a decision.

The analyzer hands it a frame when it is idle (a busy reader skips frames: the newest picture matters) and
collects finished reads on later decisions. Every result keeps the frame id and stream time it was read
from, so the scene cover can start exactly at that frame (the frames since then are still waiting in the
stream delay, see ``runtime/gate.py``). A reader error is counted, never raised: the scene model and the
person covers keep working without the HUD signal.
"""

from __future__ import annotations

import threading

import numpy as np

from streamsafe.analysis.scene import hud_hit
from streamsafe.models.base import HudReader


class HudThread:
    def __init__(self, reader: HudReader) -> None:
        self.reader = reader
        self.errors = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job: tuple[int, float, np.ndarray] | None = None
        self._busy = False
        self._done: list[tuple[int, float, str]] = []  # (frame id, stream time, keyword)
        threading.Thread(target=self._run, name="hud", daemon=True).start()

    def submit(self, frame_id: int, t: float, bgr: np.ndarray) -> bool:
        """Queue a read unless one is running or waiting. Returns whether the frame was taken."""
        with self._lock:
            if self._busy or self._job is not None:
                return False
            self._job = (frame_id, t, bgr.copy())
        self._wake.set()
        return True

    def results(self) -> list[tuple[int, float, str]]:
        """Finished reads since the last call, keyword '' when the frame showed none."""
        with self._lock:
            out, self._done = self._done, []
        return out

    def idle(self) -> bool:
        with self._lock:
            return not self._busy and self._job is None

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                job, self._job = self._job, None
                self._wake.clear()
                self._busy = job is not None
            if job is None:
                continue
            frame_id, t, bgr = job
            try:
                hit = hud_hit(self.reader(bgr))
            except Exception:  # a broken read only loses the HUD signal for this frame
                self.errors += 1
                hit = ""
            with self._lock:
                self._done.append((frame_id, t, hit))
                self._busy = False
