"""One decision for one frame:

    scene layer (explicit scene -> whole-screen cover, nothing else runs)
    -> detect people -> eye (skipped for people matched by motion) -> track ids -> risk policy
    -> pose (covered people only) -> one garment layer per covered person

Any exception produces ``Decision(ok=False)``: the renderer then fails closed for that frame.
"""

from __future__ import annotations

import time
from collections import defaultdict

import cv2
import numpy as np

from streamsafe.analysis.hud_thread import HudThread
from streamsafe.analysis.risk import RiskPolicy
from streamsafe.analysis.scene import SceneLatch
from streamsafe.analysis.tracks import Tracker
from streamsafe.config import Config
from streamsafe.garment.builder import GarmentBuilder, SafePerson, choose_mode
from streamsafe.models import Models
from streamsafe.render.motion import make_motion_ref
from streamsafe.types import ANALYSIS_SIZE, Decision, Person

SWIM_SHIFT = 0.05  # beach / pool scenes lower the threshold by up to this (x score scale)
SCENE_EVERY_S = 1.0  # scene model period when only the beach prior needs it (and on every cut)
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
        self.latch = SceneLatch(cfg.scene)
        self._scene_at = -1e9
        self._hud_at = -1e9
        self.hud = HudThread(models.hud) if models.hud is not None and cfg.scene.hud_ocr else None
        self.stats: defaultdict[str, list] = defaultdict(list)
        self.diag: list = []

    def warmup(self) -> None:
        """One pass of every model so the first real frame does not pay for lazy initialisation."""
        w, h = ANALYSIS_SIZE
        blank = np.zeros((h, w, 3), np.uint8)
        self.models.detector(blank)
        self.models.eye.embed([blank[:120, :60]])
        self.models.pose(blank, [(10, 10, 60, 120)])
        if self.models.scene is not None:
            self.models.scene(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
        if self.models.hud is not None:
            self.models.hud(blank)

    def __call__(self, frame_id: int, frame: np.ndarray, cut: bool, now: float | None = None) -> Decision:
        """``frame``: BGR, any size (resized to the analysis size; with ``full_res_crops`` a larger frame
        also supplies the eye crops)."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        im = frame if frame.shape[1::-1] == ANALYSIS_SIZE else cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        full = frame if self.cfg.eye.full_res_crops and frame.shape[1] > ANALYSIS_SIZE[0] else None
        try:
            decision = self._analyse(frame_id, im, cut, now, timings, full)
        except Exception as exc:
            decision = Decision(frame_id=frame_id, ok=False)
            self.stats["analyzer_errors"].append(f"{exc!r:.300}")
        timings["total"] = time.perf_counter() - t0
        for name, seconds in timings.items():
            self.stats["ms_" + name].append(1000 * seconds)
        self.stats["people"].append(decision.n_people)
        self.stats["risky"].append(decision.n_risky)
        return decision

    def _analyse(
        self, frame_id: int, im: np.ndarray, cut: bool, now: float | None, timings: dict[str, float], full: np.ndarray | None
    ) -> Decision:
        cfg, st = self.cfg, self.stats
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        clock = time.perf_counter()
        t = clock if now is None else now  # scene clock: the stream time offline, so runs repeat exactly

        scene_on, scene_from = self._scene_step(frame_id, rgb, im if full is None else full, cut, t)
        st["scene_on"].append(int(scene_on))
        timings["scene"], clock = _lap(clock)
        if scene_on:  # the whole frame is covered: no person work needed (saves GPU for the stream)
            return Decision(frame_id=frame_id, ok=True, scene=True, scene_from=scene_from)

        people, crops = self.detect(rgb, full, full_bgr=True)
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
        thr = self.thr - SWIM_SHIFT * self.scale * (self.p_swim if cfg.eye.scene_prior else 0.0)
        self.policy.update(people, self.tracker, thr, now)
        self.tracker.forget_old()
        timings["track"], clock = _lap(clock)

        risky = [p for p in people if p.risky]
        plan = [(p, choose_mode(p.height, self.tracker.tracks[p.tid].garment, cfg.garment.small_h)) for p in risky]
        dressed = [p for p, mode in plan if mode == "full"]
        poses = dict(zip(map(id, dressed), self.models.pose(rgb, [p.box for p in dressed]), strict=True)) if dressed else {}
        timings["pose"], clock = _lap(clock)

        safe = [SafePerson(p.mask, p.box, int(p.mask.sum())) for p in people if not p.risky and p.p < SAFE_P * self.scale]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if cfg.render.motion else None
        layers, refs = [], {}
        for person, mode in plan:
            memory = self.tracker.tracks[person.tid].garment
            layer = self.garments.build(im, person, mode, memory, safe, self.tracker.decision, cut, poses.get(id(person)))
            if layer is not None:
                if gray is not None:
                    layer.motion = make_motion_ref(gray, person.mask, person.box)
                layers.append(layer)
        if gray is not None and cfg.render.lookahead:  # a later decision may cover these people here
            covered = {layer.tid for layer in layers}
            for p in people:
                if p.tid not in covered and (ref := make_motion_ref(gray, p.mask, p.box)) is not None:
                    refs[p.tid] = ref
        timings["garment"], _ = _lap(clock)

        if cfg.output.diag_out:
            self.diag.append([frame_id, [[p.tid, list(p.box), round(p.p, 3), int(p.risky), int(p.stale)] for p in people]])
        return Decision(
            frame_id=frame_id,
            ok=True,
            layers=layers,
            n_people=len(people),
            n_risky=len(risky),
            people=[(p.tid, p.box) for p in people],
            refs=refs,
        )

    def detect(self, rgb: np.ndarray, full: np.ndarray | None = None, full_bgr: bool = False) -> tuple[list[Person], list[np.ndarray]]:
        """People in an RGB analysis frame and their eye crops (background darkened, as in head training).

        ``full``: the same frame at a higher resolution (same aspect; RGB, or BGR with ``full_bgr``); eye crops
        are then cut from it, so small people reach the eye with real detail instead of upscaled analysis pixels.
        A BGR frame is converted per crop, not as a whole."""
        people, crops = [], []
        for box, mask in self.models.detector(rgb):
            x1, y1, x2, y2 = box
            if x2 - x1 < MIN_W or y2 - y1 < self.cfg.detector.min_h or mask.sum() < MIN_MASK_PX:
                continue
            people.append(Person(box=box, mask=mask))
            if full is None:
                crops.append(eye_crop(rgb, box, mask))
            else:
                crop = eye_crop(full, box, mask, full.shape[1] / rgb.shape[1])
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if full_bgr else crop)
        return people, crops

    def _scene_step(self, frame_id: int, rgb: np.ndarray, bgr: np.ndarray, cut: bool, t: float) -> tuple[bool, int | None]:
        """Runs the scene model when due, feeds / collects the HUD thread, and returns the scene latch plus
        the earliest frame of a HUD hit that arrived now (None when the evidence is this frame)."""
        cfg, models = self.cfg, self.models
        p_explicit = None
        if models.scene is not None and (cfg.eye.scene_prior or cfg.scene.scene_guard):
            every = cfg.scene.scene_every_s if cfg.scene.scene_guard else SCENE_EVERY_S
            if cut or t - self._scene_at >= every:
                self.p_swim, p = models.scene(rgb)
                self._scene_at = t
                p_explicit = p if cfg.scene.scene_guard else None  # None: no trained head -> HUD only
        hits: list[tuple[int, float]] = []
        if self.hud is not None:
            hits = [(f, ft) for f, ft, kw in self.hud.results() if kw]
            if (cut or t - self._hud_at >= cfg.scene.hud_every_s) and self.hud.submit(frame_id, t, bgr):
                self._hud_at = t
            self.stats["hud_errors"] = [self.hud.errors]
        if not cfg.scene.scene_guard and self.hud is None:
            return False, None
        was_on = self.latch.on
        on = self.latch.update(t, p_explicit, max((ft for _, ft in hits), default=None))
        first = min((f for f, _ in hits), default=None)
        return on, (first if on and not was_on and first is not None else None)


def eye_crop(image: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """The person cut out of ``image`` with the background darkened. ``box`` / ``mask`` are in analysis
    pixels; ``scale`` maps them onto ``image`` (1.0 = the analysis frame itself)."""
    x1, y1, x2, y2 = box
    inside = mask[y1:y2, x1:x2].astype(np.uint8)
    kernel = CROP_DILATE
    if scale != 1.0:
        h, w = image.shape[:2]
        x1, y1 = round(x1 * scale), round(y1 * scale)
        x2, y2 = min(w, max(x1 + 1, round(x2 * scale))), min(h, max(y1 + 1, round(y2 * scale)))
        inside = cv2.resize(inside, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        k = round(CROP_DILATE.shape[0] * scale) | 1
        kernel = np.ones((k, k), np.uint8)
    inside = cv2.dilate(inside, kernel).astype(bool)
    crop = image[y1:y2, x1:x2].copy()
    crop[~inside] = (crop[~inside] * CROP_BACKGROUND).astype(np.uint8)  # isolate the person
    return crop


def _lap(start: float) -> tuple[float, float]:
    now = time.perf_counter()
    return now - start, now
