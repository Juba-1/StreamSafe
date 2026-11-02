"""Person identity across decisions.

A track keeps a person's appearance (eye embedding) and last mask so the same person keeps the
same id - and therefore the same risk memory and garment - from one decision to the next.

Matching (``Tracker.assign``):
  1. people already matched by motion (eye skipped) keep their proposed track;
  2. remaining people are paired greedily by score: mask overlap + appearance for tracks seen in
     the last 2 decisions, appearance only for older ones (up to 15 decisions back);
  3. unmatched people start new tracks.
A scene cut ends every track's eligibility for matching. Tracks unseen for 40 decisions are dropped.

Eye skip (``Tracker.motion_hints``): a person whose mask overlaps exactly one recently eye-checked
track (IoU >= 0.6, same height within 15 %) inherits that track's embedding and last score
instead of being embedded again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from streamsafe.types import Box, Person

APPEARANCE_MIN = 0.8  # cosine to the track's current AND first embedding
RECENT = 2  # decisions within which mask overlap counts
MATCH_WINDOW = 15  # decisions a track stays matchable
DROP_AFTER = 40  # decisions after which an unseen track is forgotten
HINT_IOU, HINT_RIVAL_IOU, HINT_HEIGHT_TOL = 0.6, 0.3, 0.15
MASK_STRIDE = 4  # masks are compared at 1/4 resolution


def box_iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + 1e-6)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    return (a & b).sum() / ((a | b).sum() + 1)


@dataclass
class RiskState:
    """Per-person risk memory, owned by ``RiskPolicy``."""

    evidence: float = 0.0  # S: EMA of log-odds above the threshold, capped
    risky: bool = False
    n_obs: int = 0
    history: list[float] = field(default_factory=list)  # last 5 eye readings
    max_p: float = 0.0
    safe_run: int = 0  # consecutive decisions with clearly safe evidence
    safe_since: float = 0.0


@dataclass
class GarmentMemory:
    """Per-person garment continuity, owned by ``GarmentBuilder``."""

    mode: str | None = None  # "small" (plain cover) or "full" (dress), with hysteresis
    keypoints: tuple[np.ndarray, np.ndarray] | None = None
    keypoints_at: int = -99
    cover: tuple[np.ndarray, Box, tuple[int, int]] | None = None  # (garment mask, box, window origin)
    cover_at: int = -99


@dataclass
class Track:
    emb: np.ndarray  # running appearance (EMA of embeddings)
    first_emb: np.ndarray
    mask4: np.ndarray  # last mask at 1/MASK_STRIDE
    box: Box
    last_seen: int  # decision index
    eye_seen: int = -99  # decision index of the last real eye reading
    risk: RiskState = field(default_factory=RiskState)
    garment: GarmentMemory = field(default_factory=GarmentMemory)

    @property
    def unit_emb(self) -> np.ndarray:
        return self.emb / np.linalg.norm(self.emb)


class Tracker:
    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self.decision = 0  # index of the current decision
        self._next_id = 0

    # -- eye skip ------------------------------------------------------------------------------
    def motion_hints(self, people: list[Person], cut: bool, eye_every: int) -> int:
        """Mark people that can reuse a track's embedding; returns how many were marked."""
        if eye_every <= 1 or cut:
            return 0
        candidates = [
            (tid, t)
            for tid, t in self.tracks.items()
            if t.last_seen == self.decision and t.risk.n_obs >= 2 and self.decision + 1 - t.eye_seen < eye_every
        ]
        taken: set[int] = set()
        for person in people:
            m4 = person.mask[::MASK_STRIDE, ::MASK_STRIDE]
            scored = sorted(
                ((_mask_iou(t.mask4, m4), tid, t) for tid, t in candidates if box_iou(t.box, person.box) >= 0.3),
                key=lambda z: -z[0],
            )
            if not scored or scored[0][0] < HINT_IOU or scored[0][1] in taken:
                continue
            if len(scored) > 1 and scored[1][0] >= HINT_RIVAL_IOU:
                continue  # two tracks fit: ambiguous, let the eye decide
            _, tid, track = scored[0]
            if abs(person.height - (track.box[3] - track.box[1])) > HINT_HEIGHT_TOL * max(person.height, 1):
                continue
            taken.add(tid)
            person.hint, person.stale = tid, True
            person.emb = track.emb / (np.linalg.norm(track.emb) + 1e-6)
            person.p = track.risk.history[-1] if track.risk.history else 0.0
        return len(taken)

    # -- identity ------------------------------------------------------------------------------
    def assign(self, people: list[Person], cut: bool) -> None:
        """Give every person a track id (``person.tid``) and refresh the tracks' geometry."""
        if cut:
            for t in self.tracks.values():
                t.last_seen = -99  # nothing is carried across a scene cut
        self.decision += 1
        masks4 = [p.mask[::MASK_STRIDE, ::MASK_STRIDE] for p in people]
        used: set[int] = set()
        claimed: set[int] = set()
        for j, person in enumerate(people):
            if person.hint is not None and person.hint in self.tracks:
                person.tid = person.hint
                used.add(j)
                claimed.add(person.hint)

        pairs = []
        for tid, track in self.tracks.items():
            age = self.decision - track.last_seen
            if tid in claimed or age > MATCH_WINDOW:
                continue
            norm = np.linalg.norm(track.emb) + 1e-6
            for j, person in enumerate(people):
                if j in used:
                    continue
                cos = float(track.emb @ person.emb / norm)
                cos_first = float(track.first_emb @ person.emb)
                if cos < APPEARANCE_MIN or cos_first < APPEARANCE_MIN:
                    continue
                if age <= RECENT:
                    overlap = _mask_iou(track.mask4, masks4[j]) if box_iou(track.box, person.box) > 0 else 0.0
                    score = overlap + 0.5 * max(0.0, cos - 0.6)
                else:
                    score = cos - 0.8 if cos > 0.85 else -1.0
                if score > 0.15:
                    pairs.append((score, tid, j))

        for _, tid, j in sorted(pairs, reverse=True):
            if tid in claimed or j in used:
                continue
            claimed.add(tid)
            used.add(j)
            people[j].tid = tid

        for j, person in enumerate(people):
            if j not in used:
                self.tracks[self._next_id] = Track(
                    emb=person.emb.copy(), first_emb=person.emb.copy(), mask4=masks4[j], box=person.box, last_seen=self.decision
                )
                person.tid = self._next_id
                self._next_id += 1

        for j, person in enumerate(people):
            track = self.tracks[person.tid]
            track.mask4, track.box, track.last_seen = masks4[j], person.box, self.decision

    def observe(self, person: Person) -> None:
        """Fold a fresh eye reading into the track's appearance (stale readings change nothing)."""
        if person.stale:
            return
        track = self.tracks[person.tid]
        track.eye_seen = self.decision
        track.emb = 0.7 * track.emb + 0.3 * person.emb

    def forget_old(self) -> None:
        for tid in [tid for tid, t in self.tracks.items() if self.decision - t.last_seen > DROP_AFTER]:
            del self.tracks[tid]
