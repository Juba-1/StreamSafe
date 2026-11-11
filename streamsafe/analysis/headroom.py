"""Realtime headroom: what this PC's model path can hold, measured once at start.

The same models take very different times on different paths (run 57, RTX 3090, one eye crop: 5 to 8 ms on
TensorRT, 19 to 26 ms on CUDA fp32), and the path can change under the streamer: TensorRT falls back to CUDA after
a failed engine build. So the engine times its own models during warm-up and derives two things from the numbers:

  * ``holds_rate``: does an ordinary decision (``USUAL_PEOPLE`` eye crops) fit into one step (100 ms at 10
    decisions/s)? If not, the engine says so at start and in the run report. It still runs: the gate fails closed
    on its own when decisions come too far apart. But it never runs below its rate silently.
  * ``budget``: the eye budget this path needs (``EYE_BUDGET_AUTO=1``, realtime only). 0 where even a full eye
    batch keeps a decision well inside the slate limit, so a fast TensorRT path runs exactly as without a budget.
    Otherwise the largest batch that does, never below ``MIN_BUDGET``.

The rule is a pure function of five numbers (``Headroom``), applied once when the analyzer starts and never
changed while it runs: no controller, nothing that can flip back and forth in mid-stream. It only ever chooses
between "no budget" and "a budget", and a budget only adds covers (``analysis/eye_schedule.py``). The offline
mode never applies it: benchmark results must not depend on the machine.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from streamsafe.config import Config
from streamsafe.models import Models
from streamsafe.types import ANALYSIS_SIZE

FULL_BATCH = 16  # the largest eye batch of one run (``MAX_BATCH`` of the eye backends)
USUAL_PEOPLE = 4  # eye crops of an ordinary decision: what the rate check assumes
# a decision may take this share of the slate limit (``max_span`` frames). The rest is for what warm-up cannot
# see: OBS encoding on the same GPU, the pose model in a crowd, the wait for the next frame
SPAN_SHARE = 0.6
# work per decision that is not a model run (resize, tracking, garments, motion references): runs 55 and 57
# measured 10 to 12 ms
CPU_MS = 12.0
MIN_BUDGET = 3  # the automatic budget never goes below this: with fewer looks a small group would stay covered for good
POSE_PEOPLE = 2
CROP = (432, 216)  # (h, w) of the timing crops: a person half a 1080p frame tall, so resizing costs what it costs live
RUNS = 3  # timed runs per measurement; the median is used


@dataclass(frozen=True)
class Headroom:
    fixed_ms: float  # detector + pose for POSE_PEOPLE + CPU_MS
    eye_one_ms: float  # one eye run with one crop
    eye_full_ms: float  # one eye run with FULL_BATCH crops
    step_ms: float  # time between two decisions at the target rate
    limit_ms: float  # the longest a decision may take (SPAN_SHARE of the slate limit)

    def decision_ms(self, crops: int) -> float:
        """Expected time of a decision that looks at ``crops`` people (the eye's time is linear between 1 and FULL_BATCH)."""
        eye = self.eye_one_ms + (self.eye_full_ms - self.eye_one_ms) * (crops - 1) / (FULL_BATCH - 1) if crops > 0 else 0.0
        return self.fixed_ms + eye

    @property
    def holds_rate(self) -> bool:
        return self.decision_ms(USUAL_PEOPLE) <= self.step_ms

    @property
    def decisions_per_s(self) -> float:
        """The rate an ordinary decision allows, at most the target rate."""
        return 1000.0 / max(self.step_ms, self.decision_ms(USUAL_PEOPLE))

    @property
    def budget(self) -> int:
        """The eye budget this path needs: 0 (none), or MIN_BUDGET .. FULL_BATCH - 1."""
        if self.decision_ms(FULL_BATCH) <= self.limit_ms:
            return 0
        fits = [n for n in range(1, FULL_BATCH) if self.decision_ms(n) <= self.limit_ms]
        return max(MIN_BUDGET, max(fits, default=0))

    def as_dict(self) -> dict:
        numbers = {k: round(v, 1) for k, v in asdict(self).items()}
        return dict(**numbers, holds_rate=self.holds_rate, decisions_per_s=round(self.decisions_per_s, 1), budget=self.budget)

    def warning(self) -> str:
        """One line for the start-up log when this path cannot hold the rate, else ""."""
        if self.holds_rate:
            return ""
        return (
            f"this GPU path holds about {self.decisions_per_s:.1f} decisions/s with {USUAL_PEOPLE} people in view "
            f"(target {1000 / self.step_ms:.0f}): one decision takes about {self.decision_ms(USUAL_PEOPLE):.0f} ms"
        )


def measure(models: Models, cfg: Config, clock: Callable[[], float] = time.perf_counter) -> Headroom:
    """Time the detector, the pose model and the eye on blank frames. Call after warm-up, so no run pays for a
    first-time setup. The models' time does not depend on what is in the picture, so blank frames measure it."""
    w, h = ANALYSIS_SIZE
    blank = np.zeros((h, w, 3), np.uint8)
    crop = np.zeros((*CROP, 3), np.uint8)

    def ms(run: Callable[[], object]) -> float:
        times = []
        for _ in range(RUNS):
            start = clock()
            run()
            times.append(1000.0 * (clock() - start))
        return sorted(times)[RUNS // 2]

    detector = ms(lambda: models.detector(blank))
    pose = ms(lambda: models.pose(blank, [(10, 10, 60, 120)] * POSE_PEOPLE))
    frame_ms = 1000.0 / cfg.runtime.fps
    return Headroom(
        fixed_ms=detector + pose + CPU_MS,
        eye_one_ms=ms(lambda: models.eye.embed([crop])),
        eye_full_ms=ms(lambda: models.eye.embed([crop] * FULL_BATCH)),
        step_ms=cfg.runtime.step_every * frame_ms,
        limit_ms=SPAN_SHARE * cfg.render.max_span * frame_ms,
    )
