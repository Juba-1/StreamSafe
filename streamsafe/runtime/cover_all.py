"""TEST ONLY: an "analyzer" that loads no model and orders the whole-screen cover for every frame.

Used by ``<exe> selftest-stream`` (``app/entry.py``) and the link tests: the complete live path runs - the SRT
listener, the delay, the output gate, the encoder, the connection to the platform, the reports - on a machine
without a graphics card or models (a CI runner). It cannot weaken the protection: with it every frame that
leaves is the full-screen cover with the sound muted, so it shows less than the product ever would, never more.
"""

from __future__ import annotations

import collections
from typing import Any

from streamsafe.config import Config
from streamsafe.types import Decision


class CoverAll:
    def __init__(self) -> None:
        self.stats: dict[str, list] = collections.defaultdict(list)
        self.diag: list = []

    def warmup(self) -> None:
        pass

    def __call__(self, frame_id: int, frame: Any, cut: bool = False, now: float | None = None) -> Decision:
        self.stats["ms_total"].append(0.1)
        return Decision(frame_id, ok=True, scene=True)


def cover_all_analyzer(cfg: Config, output_scale: float) -> CoverAll:
    """Analyzer factory (top-level: the realtime worker process imports it by name)."""
    return CoverAll()
