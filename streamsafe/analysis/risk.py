"""Turns noisy per-frame eye scores into a stable covered / not-covered state per person.

Rules (thresholds scale with the operating threshold through ``scale = thr / 0.40``):
  * evidence S = 0.85 * S + w * (logit(p) - logit(thr)), capped at S_CAP so a long-covered person
    is not stuck for many seconds; w = 0.5 when the person overlaps someone else (mixed crops)
  * p blends this frame's score with the score of the track's running appearance
  * COVER when S >= 2, or the raw score >= HARD, or (first 3 looks) p >= the first-look threshold,
    which is stricter for big close-up people (modest close-ups were the main false covers)
  * FAST CLEAR when the last two readings are very low (< 0.08 * scale) and the person never read clearly
    risky (max p < fast_clear_k * thr)
  * HANDOFF: a new track where a covered track was lost moments ago (same place) inherits its risk state
  * RELEASE otherwise only after 3 clearly-safe decisions AND release_s seconds
Eye-skipped (stale) readings repeat the last evidence; the release clock keeps running.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from streamsafe.analysis.tracks import Track, Tracker, box_iou
from streamsafe.config import RiskConfig
from streamsafe.types import Person

HARD = 0.85
S_CAP = 8.0
S_COVER = 2.0
S_SAFE = -3.0
DECAY = 0.85
SAFE_RUN_RELEASE = 3
FIRST_LOOKS = 3
OVERLAP_IOU = 0.15
HISTORY = 5
TINY_H = 20  # people shorter than this use a lower threshold
HANDOFF_MIN_P = 0.25  # x thr: below this a new track does not inherit a lost cover (run 50: palm trees, backs);
# above it a lost cover lingers ~release_s on whoever is at that place (golden rule: a moment too long, never too short)


def _logit(q: float) -> float:
    return float(np.log(max(1e-4, q) / max(1e-4, 1 - q)))


class RiskPolicy:
    def __init__(self, cfg: RiskConfig, base_thr: float, prisk: Callable[[np.ndarray], np.ndarray]) -> None:
        self.cfg = cfg
        self.base_thr = base_thr
        self.scale = base_thr / 0.40
        self.prisk = prisk

    def update(self, people: list[Person], tracker: Tracker, thr: float, now: float | None = None) -> None:
        """Set ``person.risky`` for every person of this decision (thr may include the scene prior)."""
        now = time.perf_counter() if now is None else now
        present = {p.tid for p in people}
        for person in people:
            track = tracker.tracks[person.tid]
            if self.cfg.handoff and track.risk.n_obs == 0:
                self._handoff(person, track, tracker, present)
            self._observe(person, track)
            tracker.observe(person)
            person.risky = self._decide(person, track, people, tracker, thr, now)

    def _handoff(self, person: Person, track: Track, tracker: Tracker, present: set[int]) -> None:
        """Copy the risk state of a covered track lost moments ago at this place (identity broke, person did not)."""
        if person.p < HANDOFF_MIN_P * self.base_thr:  # a very clear "no" (a tree, a wall, a man's back) never inherits
            return
        best, best_iou = None, self.cfg.handoff_iou
        for tid, old in tracker.tracks.items():
            if tid in present or not old.risk.risky or tracker.decision - old.last_seen > self.cfg.handoff_window:
                continue
            if old.last_seen < 0:  # cleared by a scene cut
                continue
            h_old = old.box[3] - old.box[1]
            if not 0.7 <= person.height / max(1, h_old) <= 1.4:  # not the same body size: someone else
                continue
            iou = box_iou(old.box, person.box)
            if iou >= best_iou:
                best, best_iou = old, iou
        if best is not None:
            r, o = track.risk, best.risk
            r.risky, r.evidence, r.max_p, r.history = True, o.evidence, o.max_p, list(o.history)
            r.n_obs = FIRST_LOOKS  # past the first-look rule: it is the same person
            o.risky = False  # one cover per person: the old track is gone

    @staticmethod
    def _observe(person: Person, track: Track) -> None:
        r = track.risk
        r.n_obs += 1
        if person.stale:
            person.p = r.history[-1] if r.history else person.p
        else:
            r.history = [*r.history, person.p][-HISTORY:]
            r.max_p = max(r.max_p, person.p)

    def _decide(self, person: Person, track: Track, people: list[Person], tracker: Tracker, thr: float, now: float) -> bool:
        cfg, scale, r = self.cfg, self.scale, track.risk
        p_track = float(self.prisk(track.unit_emb[None])[0])
        p = 0.5 * person.p + 0.5 * p_track

        others = [o for o in people if o is not person]
        overlap = max([box_iou(person.box, o.box) for o in others] or [0])
        overlap_risky = max([box_iou(person.box, o.box) for o in others if tracker.tracks[o.tid].risk.risky] or [0])
        weight = 0.5 if overlap > OVERLAP_IOU else 1.0
        th = thr if person.height >= TINY_H else min(thr, 0.2 * scale)

        r.evidence = min(S_CAP, DECAY * r.evidence + weight * (_logit(p) - _logit(th)))
        r.safe_run = r.safe_run + 1 if r.evidence < S_SAFE else 0
        if r.safe_run == 1:
            r.safe_since = now

        prior = r.history[:-1]
        first_look = max(
            th,
            cfg.young_k * self.base_thr if person.height >= cfg.young_min_h else self.base_thr,
            # someone half-hidden behind a covered person: only a clear reading counts
            0.6 * scale if (overlap_risky > OVERLAP_IOU and prior and max(prior) < 0.2 * scale) else 0,
        )
        never_clear = HARD if cfg.fast_clear_k <= 0 else min(HARD, cfg.fast_clear_k * thr)
        fast_clear = len(r.history) >= 2 and max(r.history[-2:]) < 0.08 * scale and overlap <= OVERLAP_IOU and r.max_p < never_clear

        if fast_clear and r.risky:
            r.risky = False
            r.evidence = min(r.evidence, 0)
        elif r.evidence >= S_COVER or person.p >= HARD or (r.n_obs <= FIRST_LOOKS and p >= first_look):
            r.risky = True
            r.safe_run = 0
        elif r.risky and r.safe_run >= SAFE_RUN_RELEASE and now - r.safe_since >= cfg.release_s:
            r.risky = False
        return r.risky
