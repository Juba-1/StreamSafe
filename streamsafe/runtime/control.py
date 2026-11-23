"""What the streamer changes while live, without stopping the stream: one small JSON file the window writes and
the engine re-reads (``CONTROL_FILE``). Missing or unreadable: the defaults, which change nothing.

    {"panic": false, "everyone": false, "thr_scale": 1.4, "zones": [[x0, y0, x1, y1], ...]}

``zones``: parts of the picture (fractions 0-1, e.g. the streamer's webcam) where people are never covered by
person covers; panic and whole-screen covers still cover everything.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Live:
    panic: bool = False
    everyone: bool = False  # dress every person, men too (ads): the brand garment on everyone
    thr_scale: float = 0.0  # 0 = keep the start value
    zones: tuple[tuple[float, float, float, float], ...] = field(default_factory=tuple)

    def in_zone(self, x: float, y: float) -> bool:
        """(x, y) as fractions of the picture."""
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in self.zones)


def parse(data: object) -> Live:
    if not isinstance(data, dict):
        return Live()
    zones = []
    for z in data.get("zones", []) or []:
        try:
            x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in z)
        except (TypeError, ValueError):
            continue
        if x1 > x0 and y1 > y0:
            zones.append((x0, y0, x1, y1))
    try:
        k = float(data.get("thr_scale", 0) or 0)
    except (TypeError, ValueError):
        k = 0.0
    return Live(
        panic=bool(data.get("panic", False)),
        everyone=bool(data.get("everyone", False)),
        thr_scale=k if 0.3 <= k <= 3.0 else 0.0,
        zones=tuple(zones),
    )


class LiveControl:
    """Re-reads the file at most every ``every_s`` and only when it changed."""

    def __init__(self, path: str | os.PathLike[str] | None, every_s: float = 0.2) -> None:
        self.path = Path(path) if path else None
        self.every_s = every_s
        self._at = -1e9
        self._mtime = -1.0
        self.now = Live()

    @classmethod
    def from_env(cls) -> LiveControl:
        return cls(os.environ.get("CONTROL_FILE") or None)

    def get(self) -> Live:
        if self.path is None:
            return self.now
        t = time.monotonic()
        if t - self._at < self.every_s:
            return self.now
        self._at = t
        try:
            m = self.path.stat().st_mtime
        except OSError:
            self.now, self._mtime = Live(), -1.0
            return self.now
        if m != self._mtime:
            try:
                self.now = parse(json.loads(self.path.read_text(encoding="utf-8")))
                self._mtime = m
            except (OSError, ValueError):
                pass  # half-written: the next read gets it
        return self.now
