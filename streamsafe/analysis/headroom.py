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

The start rule is a pure function of five numbers (``Headroom``), applied once when the analyzer starts. It only
ever chooses between "no budget" and "a budget", and a budget only adds covers (``analysis/eye_schedule.py``).
The offline mode never applies it: benchmark results must not depend on the machine.

What the start cannot know is what else will use the GPU once the stream runs (the game, the encoder, a laptop
that throttles when it gets hot). ``BudgetGovernor`` (``EYE_BUDGET_ADAPT``, realtime only) follows the decisions'
own measured times: one that ran over the limit lowers the budget at once, a long calm stretch raises it by one,
never above where it started. Like the budget itself it can only add covers, never remove one.
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


GOVERNOR_FLOOR = 1  # the governor may go below MIN_BUDGET: one look per decision still gets everybody seen in turn
CALM_DECISIONS = 30  # decisions at the full budget, all with room for one more crop, before the budget is raised by one
RELAX_SHARE = 0.85  # "room for one more crop": the decision scaled to budget + 1 crops stays under this share of the limit
PLAIN_DECISIONS = 100  # plain covers stay on this long (10 s) after the slow decision that turned them on ...
PLAIN_MAX = 1200  # ... doubling whenever dresses prove too slow again right after coming back, up to this (2 min)


class BudgetGovernor:
    """What the analyzer may spend per decision while the stream runs (realtime only). Deterministic for a given
    series of decision times; both of its levers can only make the cover bigger, never smaller.

    The eye budget:
    * A decision that looked at ``crops`` people and took longer than ``limit_ms`` lowers the budget, in
      proportion to how far over it was and by at least one, never below ``GOVERNOR_FLOOR``. One slow decision is
      enough: the next one over the slate limit would be a slate for everybody.
    * The budget is raised by one only after ``CALM_DECISIONS`` decisions that used the whole budget and still
      had room for one more crop; a slow decision in between starts the count again. Decisions that did not need
      the budget prove nothing about a bigger one and are not counted.
    * It never goes above the start value (``start`` 0 = no limit: back to no limit after ``FULL_BATCH - 1``).

    Plain covers (``plain``): a slow decision whose time went into drawing dresses (pose model + cloth, CPU work
    the eye budget cannot touch) more than into the eye switches every garment to the plain full-body cover for
    ``PLAIN_DECISIONS`` decisions. The plain cover hides more than the dress (the whole body, head included) and
    costs a tenth of it. If dresses are too slow again within ``PLAIN_DECISIONS`` of coming back, the next plain
    period is twice as long (at most ``PLAIN_MAX``), so a PC that cannot afford dresses does not flip all the time.
    """

    def __init__(self, start: int, limit_ms: float) -> None:
        self.start = start
        self.limit_ms = limit_ms
        self.budget = start
        self.plain = False
        self._calm = 0
        self._plain_left = 0
        self._plain_period = PLAIN_DECISIONS
        self._since_plain = PLAIN_DECISIONS  # decisions since plain covers were last switched off
        self.lowered = 0  # how often it had to lower the budget (run report)
        self.lowest = start  # the smallest budget it came down to (0 = it never needed one)
        self.plain_periods = 0  # how often it switched to plain covers (run report)

    def update(self, total_ms: float, crops: int, eye_ms: float = 0.0, dress_ms: float = 0.0) -> int:
        """Takes the time of the decision just made, its eye batch, and the time its eye and its dresses (pose +
        garments) took; returns the eye budget of the next decision and sets ``plain`` for it."""
        current = self.budget or FULL_BATCH
        slow = total_ms > self.limit_ms
        if self.plain:
            self._plain_left -= 1
            if slow:  # still slow without dresses: stay plain for a full period from here
                self._plain_left = self._plain_period
            elif self._plain_left <= 0:
                self.plain, self._since_plain = False, 0
        else:
            self._since_plain += 1
            if slow and dress_ms > eye_ms:
                if self._since_plain <= PLAIN_DECISIONS:
                    self._plain_period = min(PLAIN_MAX, 2 * self._plain_period)
                else:
                    self._plain_period = PLAIN_DECISIONS
                self.plain, self._plain_left, self.plain_periods = True, self._plain_period, self.plain_periods + 1
        if slow:
            self._calm = 0
            if crops > 0:
                fits = int(min(current, crops) * self.limit_ms / total_ms)
                lower = max(GOVERNOR_FLOOR, min(current - 1, fits))
                if lower != self.budget:
                    self.budget, self.lowered = lower, self.lowered + 1
                    self.lowest = lower if self.lowest == 0 else min(self.lowest, lower)
        elif self.budget != self.start and crops >= current:
            if total_ms * (current + 1) / current <= RELAX_SHARE * self.limit_ms:
                self._calm += 1
                if self._calm >= CALM_DECISIONS:
                    self._calm = 0
                    raised = current + 1
                    self.budget = self.start if raised >= (self.start or FULL_BATCH) else raised
            else:
                self._calm = 0
        return self.budget
