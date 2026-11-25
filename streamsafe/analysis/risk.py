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
Score-only strengths (balanced, light; ``RiskPolicy.light``): the score alone decides - no lower bar for tiny
people, no stricter first look, half the release time, a lower evidence cap - and far-away people need FAR_THR.
Eye-skipped (stale) readings repeat the last evidence; the release clock keeps running.
A person whose look the eye budget put off (``analysis/eye_schedule.py``) is covered and nothing else happens:
no evidence, no observation counted, and the release clock stands still. When the look comes, evidence for
covering counts once for every decision they waited, so the cover is not later than without the wait.
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
# score-only strengths (balanced, light): someone small and far away needs a near-certain reading. The eye misreads
# small crops: far-away men on bikes read 0.83-0.89 on the trailer (runs/trailer_scores_r3), clear swimwear 0.9+
FAR_H = 150  # analysis px (of 540; as RiskConfig.young_min_h): shorter than this counts as far away
FAR_THR = 0.9
S_CAP_LIGHT = 4.0  # score-only: a lower cap, so a cover ends sooner once the readings drop
MAX_WAITED = 30  # decisions of waiting that count at the look (0.85 ** 30 < 0.01: more changes nothing)
HANDOFF_MIN_P = 0.25  # x thr: below this a new track does not inherit a lost cover (run 50: palm trees, backs);
# above it a lost cover lingers ~release_s on whoever is at that place (golden rule: a moment too long, never too short)


def _logit(q: float) -> float:
    return float(np.log(max(1e-4, q) / max(1e-4, 1 - q)))


class RiskPolicy:
    def __init__(self, cfg: RiskConfig, base_thr: float, prisk: Callable[[np.ndarray], np.ndarray]) -> None:
        self.cfg = cfg
        self.base_thr = base_thr
        self.light = False  # set by the sensitivity slider (Analyzer.set_sensitivity)
        self.scale = base_thr / 0.40
        self.prisk = prisk
        self._last_update: float | None = None  # time of the previous decision

    def update(self, people: list[Person], tracker: Tracker, thr: float, now: float | None = None) -> None:
        """Set ``person.risky`` for every person of this decision (thr may include the scene prior)."""
        now = time.perf_counter() if now is None else now
        since_last = 0.0 if self._last_update is None else now - self._last_update
        self._last_update = now
        present = {p.tid for p in people}
        for person in people:
            track = tracker.tracks[person.tid]
            if person.put_off:  # not seen this time: covered, and this decision never counts toward a release
                if track.risk.safe_run:
                    track.risk.safe_since += since_last
                # "light" (the default): someone the eye skipped this time keeps their last verdict (a man stays
                # uncovered); someone it has never looked at is not covered yet. Stricter settings cover them.
                person.risky = (track.risk.risky if track.risk.n_obs else False) if self.light else True
                continue
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

    def _bar(self, height: int, thr: float) -> float:
        """The score this person must read: score-only asks far-away people for near certainty; the careful
        rules ask tiny people for less (they are hard to read and rarely shown long)."""
        if self.light:
            return max(thr, FAR_THR) if height < FAR_H else thr
        return thr if height >= TINY_H else min(thr, 0.2 * self.scale)

    def _decide(self, person: Person, track: Track, people: list[Person], tracker: Tracker, thr: float, now: float) -> bool:
        cfg, scale, r = self.cfg, self.scale, track.risk
        p_track = float(self.prisk(track.unit_emb[None])[0])
        p = 0.5 * person.p + 0.5 * p_track

        others = [o for o in people if o is not person]
        overlap = max([box_iou(person.box, o.box) for o in others] or [0])
        overlap_risky = max([box_iou(person.box, o.box) for o in others if tracker.tracks[o.tid].risk.risky] or [0])
        weight = 0.5 if overlap > OVERLAP_IOU else 1.0
        # "light" sensitivity (the streamer's slider): only the score decides - no lower bar for tiny people, no
        # early first-look cover, a shorter hold. Measured 23 Nov: a quarter of covers came from these rules
        light = self.light
        th = self._bar(person.height, thr)
        hard = max(HARD, th)
        cap = S_CAP_LIGHT if light else S_CAP

        gain = weight * (_logit(p) - _logit(th))
        # an uncovered person who waited for this look: the decisions they waited count with the reading they
        # get now, as they would have with a look in time. Only toward covering - a low reading counts once
        looks = 1 + min(person.waited, MAX_WAITED) if gain > 0 and not r.risky else 1
        for _ in range(looks):
            r.evidence = min(cap, DECAY * r.evidence + gain)
        r.safe_run = r.safe_run + 1 if r.evidence < S_SAFE else 0
        if r.safe_run == 1:
            r.safe_since = now

        prior = r.history[:-1]
        first_look = (
            th
            if light
            else max(
                th,
                cfg.young_k * self.base_thr if person.height >= cfg.young_min_h else self.base_thr,
                # someone half-hidden behind a covered person: only a clear reading counts
                0.6 * scale if (overlap_risky > OVERLAP_IOU and prior and max(prior) < 0.2 * scale) else 0,
            )
        )
        never_clear = HARD if cfg.fast_clear_k <= 0 else min(HARD, cfg.fast_clear_k * thr)
        fast_clear = len(r.history) >= 2 and max(r.history[-2:]) < 0.08 * scale and overlap <= OVERLAP_IOU and r.max_p < never_clear

        if fast_clear and r.risky:
            r.risky = False
            r.evidence = min(r.evidence, 0)
        elif r.evidence >= S_COVER or person.p >= hard or (r.n_obs <= FIRST_LOOKS and p >= first_look):
            r.risky = True
            r.safe_run = 0
        elif r.risky and r.safe_run >= SAFE_RUN_RELEASE and now - r.safe_since >= cfg.release_s * (0.5 if light else 1.0):
            r.risky = False
        return r.risky
