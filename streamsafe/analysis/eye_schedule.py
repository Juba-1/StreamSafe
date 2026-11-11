"""Which people the eye looks at in one decision.

The eye is the engine's most expensive model and its cost grows with every person it looks at. A crowd is slow
for two reasons: after a cut everybody is new and needs a look at once, and people followed by motion fall due
together every ``eye_every`` decisions. A decision that takes too long is a slate for everybody (``max_span``),
so with a budget (``eye_budget`` crops per decision) the crowd's looks are spread over several decisions.

Without a budget (0, the default) nothing here changes anything: everybody the tracker cannot carry by motion
is looked at now, however many that is.

With a budget:
  * The biggest people are looked at first: they are what viewers see most. Covered people whose look falls due
    come last (waiting changes nothing on screen for them), until they have waited ``eye_max_age`` decisions.
  * Whoever needs a look and does not get one is PUT OFF and covered until the eye has seen them:
      - followed by motion (``Tracker.motion_matches``): keeps the track; the risk policy leaves its state alone;
      - nobody the tracker knows (new, moved a lot, after a cut): a plain cover under a waiting id, no track yet.
    Nobody is shown on the strength of a look that did not happen, and no cover ends without a look.
  * ``Person.waited`` tells the risk policy how long a person was put off before the look they get now, so
    their cover comes no later than it would have without the wait.
  * Spare budget goes to EARLY looks at uncovered people with the oldest reading, so a crowd's looks do not all
    fall due in the same decision. Covered people are never looked at early: releases keep their pace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from streamsafe.analysis.tracks import MASK_STRIDE, Tracker, mask_iou
from streamsafe.config import EyeConfig
from streamsafe.types import Person

WAIT_IOU = 0.3  # mask overlap that makes a waiting person the one who waited at that place in the last decision
FIRST_WAITING_ID = -2  # waiting ids count down from here (-1 is "no track yet" in ``Person``)


@dataclass
class Looks:
    look: list[int]  # indices into ``people``: the crops the eye gets now
    waiting: list[int] = field(default_factory=list)  # put off without a track: covered under ``person.tid`` < 0
    skipped: int = 0  # followed by motion with a recent reading: no look needed
    early: int = 0
    put_off: int = 0


class EyeScheduler:
    def __init__(self, cfg: EyeConfig) -> None:
        self.cfg = cfg
        # crops per decision, 0 = no limit. Set once before the first decision (the config, or the analyzer's
        # warm-up with ``eye_budget_auto``) and not changed afterwards
        self.budget = cfg.eye_budget
        self._owed: dict[int, int] = {}  # track id -> decisions its look has been put off so far
        self._waiting: list[tuple[int, np.ndarray, int]] = []  # people without a track: (waiting id, mask / 4, decisions waited)
        self._next_id = FIRST_WAITING_ID
        # decisions in a row in which somebody other than a covered person was put off: nobody can have waited
        # longer, so it stands in for the wait of a person whose identity was lost on the way
        self._backlog = 0

    def plan(self, people: list[Person], tracker: Tracker, cut: bool) -> Looks:
        """Marks every person as looked at now (untouched), carried by motion (``stale``) or put off
        (``put_off``), and returns the indices to embed."""
        cfg, budget = self.cfg, self.budget
        skipped = tracker.motion_hints(people, cut, cfg.eye_every)
        need = [j for j, p in enumerate(people) if not p.stale]
        if budget <= 0:
            return Looks(need, skipped=skipped)
        if cut:
            self._waiting = []  # places mean nothing across a cut
        known = {} if cut else tracker.motion_matches(people)  # person index -> track that is clearly this person

        def urgency(j: int) -> tuple[int, int, int]:
            track = tracker.tracks.get(known.get(j, -1))
            if track is not None and track.risk.risky:
                age = tracker.decision + 1 - track.eye_seen
                if age < cfg.eye_max_age:
                    return 1, -age, -people[j].height
            return 0, -people[j].height, int(track is None)

        order = sorted(need, key=urgency)
        look, late = order[:budget], order[budget:]

        # early looks: up to the crowd's average need per decision, never more than the budget
        early: list[int] = []
        target = min(budget, math.ceil(len(people) / max(1, cfg.eye_every)))
        if len(look) < target:
            idle = [j for j, p in enumerate(people) if p.stale and not tracker.tracks[p.hint].risk.risky]
            idle.sort(key=lambda j: (tracker.tracks[people[j].hint].eye_seen, -people[j].height))
            early = idle[: target - len(look)]
            for j in early:
                p = people[j]
                known[j] = p.hint
                p.hint, p.stale, p.emb, p.p = None, False, None, 0.0

        claimed: set[int] = set()
        owed: dict[int, int] = {}
        waiting: list[tuple[int, np.ndarray, int]] = []
        out = Looks(sorted(look + early), skipped=skipped - len(early), early=len(early), put_off=len(late))
        for j in out.look:
            tid = known.get(j)
            people[j].waited = self._owed.get(tid, 0) if tid is not None else self._waited(people[j], claimed)[1]
        for j in late:
            person, tid = people[j], known.get(j)
            person.put_off = True
            if tid is not None:
                tracker.carry(person, tid)
                owed[tid] = self._owed.get(tid, 0) + 1
            else:
                wid, waited = self._waited(person, claimed)
                if wid is None:
                    wid, self._next_id = self._next_id, self._next_id - 1
                person.tid, person.stale, person.risky, person.p = wid, True, True, 1.0  # not seen: assumed risky
                waiting.append((wid, person.mask[::MASK_STRIDE, ::MASK_STRIDE], waited + 1))
                out.waiting.append(j)
        self._backlog = self._backlog + 1 if any(urgency(j)[0] == 0 for j in late) else 0
        self._owed, self._waiting = owed, waiting
        return out

    def _waited(self, person: Person, claimed: set[int]) -> tuple[int | None, int]:
        """(waiting id, decisions waited) of the person who waited at this place in the last decision; without
        one (None), the longest anybody can have waited."""
        m4 = person.mask[::MASK_STRIDE, ::MASK_STRIDE]
        best, best_iou = None, WAIT_IOU
        for wid, mask4, waited in self._waiting:
            iou = mask_iou(mask4, m4)
            if wid not in claimed and iou >= best_iou:
                best, best_iou = (wid, waited), iou
        if best is None:
            return None, self._backlog
        claimed.add(best[0])
        return best
