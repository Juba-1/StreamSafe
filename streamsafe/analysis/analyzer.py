"""One decision for one frame:

    detect people -> eye (skipped for people matched by motion) -> track ids -> risk policy
    -> pose (covered people only) -> one garment layer per covered person

Any exception produces ``Decision(ok=False)``: the renderer then fails closed for that frame.
"""

from __future__ import annotations

import time
from collections import defaultdict

import cv2
import numpy as np

from streamsafe.analysis.risk import RiskPolicy
from streamsafe.analysis.tracks import Tracker
from streamsafe.config import Config
from streamsafe.garment.builder import GarmentBuilder, SafePerson, choose_mode
from streamsafe.models import Models
from streamsafe.render.motion import make_motion_ref
from streamsafe.types import ANALYSIS_SIZE, Decision, Person

SWIM_SHIFT = 0.05  # beach / pool scenes lower the threshold by up to this (x score scale)
SCENE_EVERY_S = 1.0  # scene prior refresh period (and on every cut)
MIN_W = 12
MIN_MASK_PX = 200
CROP_DILATE = np.ones((9, 9), np.uint8)
CROP_BACKGROUND = 0.18  # background brightness in eye crops, as in head training
SAFE_P = 0.2  # people below this (x scale) are "clearly safe": garments never spill onto them


class Analyzer:
    def __init__(self, cfg: Config, models: Models, output_scale: float) -> None:
        self.cfg = cfg
        self.models = models
        self.thr = cfg.risk.thr or models.eye.thr
        self.scale = self.thr / 0.40
        self.tracker = Tracker()
        self.policy = RiskPolicy(cfg.risk, self.thr, models.eye.prisk)
        self.garments = GarmentBuilder(cfg.garment, output_scale)
        self.p_swim = 0.0
        self._scene_at = -1e9
        self.stats: defaultdict[str, list] = defaultdict(list)
        self.diag: list = []

    def warmup(self) -> None:
        """One pass of every model so the first real frame does not pay for lazy initialisation."""
        w, h = ANALYSIS_SIZE
        blank = np.zeros((h, w, 3), np.uint8)
        self.models.detector(blank)
        self.models.eye.embed([blank[:120, :60]])
        self.models.pose(blank, [(10, 10, 60, 120)])

    def __call__(self, frame_id: int, frame: np.ndarray, cut: bool, now: float | None = None) -> Decision:
        """``frame``: BGR, any size (resized to the analysis size)."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        im = frame if frame.shape[1::-1] == ANALYSIS_SIZE else cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        try:
            decision = self._analyse(frame_id, im, cut, now, timings)
        except Exception as exc:
            decision = Decision(frame_id=frame_id, ok=False)
            self.stats["analyzer_errors"].append(f"{exc!r:.300}")
        timings["total"] = time.perf_counter() - t0
        for name, seconds in timings.items():
            self.stats["ms_" + name].append(1000 * seconds)
        self.stats["people"].append(decision.n_people)
        self.stats["risky"].append(decision.n_risky)
        return decision

    def _analyse(self, frame_id: int, im: np.ndarray, cut: bool, now: float | None, timings: dict[str, float]) -> Decision:
        cfg, st = self.cfg, self.stats
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        clock = time.perf_counter()

        people, crops = self._detect(rgb)
        timings["detect"], clock = _lap(clock)

        st["eye_skipped"].append(self.tracker.motion_hints(people, cut, cfg.eye.eye_every))
        need = [j for j, p in enumerate(people) if not p.stale]
        if need:
            emb = self.models.eye.embed([crops[j] for j in need])
            for j, e, p in zip(need, emb, self.models.eye.prisk(emb), strict=True):
                people[j].emb, people[j].p = e, float(p)
        st["embedded"].append(len(need))
        timings["eye"], clock = _lap(clock)

        self.tracker.assign(people, cut)
        thr = self.thr - SWIM_SHIFT * self.scale * self._swim_prior(rgb, cut)
        self.policy.update(people, self.tracker, thr, now)
        self.tracker.forget_old()
        timings["track"], clock = _lap(clock)

        risky = [p for p in people if p.risky]
        plan = [(p, choose_mode(p.height, self.tracker.tracks[p.tid].garment, cfg.garment.small_h)) for p in risky]
        full = [p for p, mode in plan if mode == "full"]
        poses = dict(zip(map(id, full), self.models.pose(rgb, [p.box for p in full]), strict=True)) if full else {}
        timings["pose"], clock = _lap(clock)

        safe = [SafePerson(p.mask, p.box, int(p.mask.sum())) for p in people if not p.risky and p.p < SAFE_P * self.scale]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if cfg.render.motion else None
        layers = []
        for person, mode in plan:
            memory = self.tracker.tracks[person.tid].garment
            layer = self.garments.build(im, person, mode, memory, safe, self.tracker.decision, cut, poses.get(id(person)))
            if layer is not None:
                if gray is not None:
                    layer.motion = make_motion_ref(gray, person.mask, person.box)
                layers.append(layer)
        timings["garment"], _ = _lap(clock)

        if cfg.output.diag_out:
            self.diag.append([frame_id, [[p.tid, list(p.box), round(p.p, 3), int(p.risky), int(p.stale)] for p in people]])
        return Decision(frame_id=frame_id, ok=True, layers=layers, n_people=len(people), n_risky=len(risky))

    def _detect(self, rgb: np.ndarray) -> tuple[list[Person], list[np.ndarray]]:
        people, crops = [], []
        for box, mask in self.models.detector(rgb):
            x1, y1, x2, y2 = box
            if x2 - x1 < MIN_W or y2 - y1 < self.cfg.detector.min_h or mask.sum() < MIN_MASK_PX:
                continue
            inside = cv2.dilate(mask[y1:y2, x1:x2].astype(np.uint8), CROP_DILATE).astype(bool)
            crop = rgb[y1:y2, x1:x2].copy()
            crop[~inside] = (crop[~inside] * CROP_BACKGROUND).astype(np.uint8)  # isolate the person
            people.append(Person(box=box, mask=mask))
            crops.append(crop)
        return people, crops

    def _swim_prior(self, rgb: np.ndarray, cut: bool) -> float:
        scene = self.models.scene
        if scene is None or not self.cfg.eye.scene_prior:
            return 0.0
        if cut or time.perf_counter() - self._scene_at > SCENE_EVERY_S:
            self.p_swim = scene.pswim(rgb)
            self._scene_at = time.perf_counter()
        return self.p_swim


def _lap(start: float) -> tuple[float, float]:
    now = time.perf_counter()
    return now - start, now
