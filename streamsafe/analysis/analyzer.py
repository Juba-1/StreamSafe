"""One decision for one frame:

    scene layer (explicit scene -> whole-screen cover, nothing else runs)
    -> detect people -> eye (skipped for people matched by motion; with an eye budget, put off for some: they
       are covered until their look, ``analysis/eye_schedule.py``) -> track ids -> risk policy
    -> pose (covered people only) -> one garment layer per covered person

Any exception produces ``Decision(ok=False)``: the renderer then fails closed for that frame.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from streamsafe import sensitivity
from streamsafe.analysis import body_rule, headroom
from streamsafe.analysis.eye_schedule import EyeScheduler
from streamsafe.analysis.hud_thread import HudThread
from streamsafe.analysis.risk import S_COVER, RiskPolicy
from streamsafe.analysis.scene import SceneLatch
from streamsafe.analysis.tracks import GarmentMemory, Tracker, box_iou
from streamsafe.config import Config
from streamsafe.garment.builder import GarmentBuilder, SafePerson, choose_mode
from streamsafe.models import Models
from streamsafe.render.motion import make_motion_ref
from streamsafe.runtime.control import LiveControl
from streamsafe.runtime.threads import usable_cpus
from streamsafe.types import ANALYSIS_SIZE, Decision, Layer, Person

SWIM_SHIFT = 0.05  # beach / pool scenes lower the threshold by up to this (x score scale)
SCENE_EVERY_S = 1.0  # scene model period when only the beach prior needs it (and on every cut)
MIN_W = 12
MIN_MASK_PX = 200
CROP_DILATE = np.ones((9, 9), np.uint8)
CROP_BACKGROUND = 0.18  # background brightness in eye crops, as in head training
COAST_FREE_IOU = 0.1  # a detection overlapping a lost person's last box this much ends coasting there
# only big (close) people coast: that is where a detector miss shows a lot and happens (bent over, half out of
# the frame). Mid-size and small people coasting made floating garments in fast scenes (run 52 volleyball)
COAST_MIN_H = 0.35  # x frame height
SAFE_P = 0.2  # people below this (x scale) are "clearly safe": garments never spill onto them
# every batch size the eye and the pose model can meet is run once at start: ONNX Runtime, cuDNN and DirectML
# prepare kernels per input shape the first time they see it, and that pause in a live crowd would be a slate
WARM_EYE, WARM_POSE = 16, 8


class Analyzer:
    def __init__(self, cfg: Config, models: Models, output_scale: float) -> None:
        self.cfg = cfg
        self.models = models
        self._base_thr = cfg.risk.thr or models.eye.thr
        self._k = cfg.risk.thr_scale
        self.thr = min(0.95, self._base_thr * self._k)
        self.scale = self.thr / 0.40
        self.live = LiveControl.from_env()  # the streamer's sensitivity slider, changed while live
        self.tracker = Tracker()
        self.eyes = EyeScheduler(cfg.eye)
        self.policy = RiskPolicy(cfg.risk, self.thr, models.eye.prisk)
        self.policy.light = self._k >= sensitivity.SCORE_ONLY_K
        self.garments = GarmentBuilder(cfg.garment, output_scale)
        self.p_swim = 0.0
        self.latch = SceneLatch(cfg.scene)
        self._scene_at = -1e9
        self._hud_at = -1e9
        self.hud = HudThread(models.hud) if models.hud is not None and cfg.scene.hud_ocr else None
        self.stats: defaultdict[str, list] = defaultdict(list)
        self.diag: list = []
        self.headroom: headroom.Headroom | None = None  # this PC's measured model times, set by ``warmup``
        self.clock: Callable[[], float] = time.perf_counter  # what decisions are timed with (tests: a clock of their own)
        self.governor: headroom.BudgetGovernor | None = None  # realtime: follows the measured decision times
        threads = cfg.garment.garment_threads or min(4, max(1, usable_cpus() // 2))
        self._pool = ThreadPoolExecutor(threads, thread_name_prefix="garment") if threads > 1 else None

    def warmup(self, clock: Callable[[], float] = time.perf_counter) -> None:
        """One pass of every model so the first real frame does not pay for lazy initialisation, then the
        model times of this PC (``analysis/headroom.py``): they go into the run report, and with
        ``eye_budget_auto`` they set the eye budget, once, before the first decision."""
        w, h = ANALYSIS_SIZE
        self.clock = clock
        blank = np.zeros((h, w, 3), np.uint8)
        self.models.detector(blank)
        for n in range(1, min(WARM_EYE, self.eyes.budget or WARM_EYE) + 1):
            self.models.eye.embed([blank[:120, :60]] * n)
        for n in range(1, WARM_POSE + 1):
            self.models.pose(blank, [(10, 10, 60, 120)] * n)
        if self.models.scene is not None:
            self.models.scene(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
        if self.models.hud is not None:
            self.models.hud(blank)
        self.headroom = headroom.measure(self.models, self.cfg, clock)
        report = self.headroom.as_dict()
        # an explicit EYE_BUDGET wins; offline never follows the clock (the same result on any machine)
        if self.cfg.eye.eye_budget_auto and self.eyes.budget == 0 and self.cfg.runtime.mode == "realtime":
            self.eyes.budget = self.headroom.budget
            report["budget_applied"] = self.eyes.budget
        if self.cfg.eye.eye_budget_adapt and self.cfg.runtime.mode == "realtime":
            self.governor = headroom.BudgetGovernor(self.eyes.budget, self.headroom.limit_ms)
        self.stats["headroom"] = [report]

    def set_sensitivity(self, k: float) -> None:
        """k > 1: only clearer cases are covered; k < 1: more is covered. Takes effect on the next decision."""
        self._k = k
        self.thr = min(0.95, self._base_thr * k)
        self.scale = self.thr / 0.40
        self.policy.base_thr, self.policy.scale = self.thr, self.thr / 0.40
        self.policy.light = k >= sensitivity.SCORE_ONLY_K

    def __call__(self, frame_id: int, frame: np.ndarray, cut: bool, now: float | None = None) -> Decision:
        """``frame``: BGR, any size (resized to the analysis size; with ``full_res_crops`` a larger frame
        also supplies the eye crops)."""
        k = self.live.get().thr_scale
        if k and k != self._k:
            self.set_sensitivity(k)
        t0 = self.clock()
        timings: dict[str, float] = {}
        im = frame if frame.shape[1::-1] == ANALYSIS_SIZE else cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        full = frame if self.cfg.eye.full_res_crops and frame.shape[1] > ANALYSIS_SIZE[0] else None
        try:
            decision = self._analyse(frame_id, im, cut, now, timings, full)
        except Exception as exc:
            decision = Decision(frame_id=frame_id, ok=False)
            self.stats["analyzer_errors"].append(f"{exc!r:.300}")
        timings["total"] = self.clock() - t0
        for name, seconds in timings.items():
            self.stats["ms_" + name].append(1000 * seconds)
        if self.governor is not None:  # what the next decision may spend follows what this one cost
            looked = decision.ok and not decision.scene and "eye" in timings
            self.eyes.budget = self.governor.update(
                1000 * timings["total"],
                self.stats["embedded"][-1] if looked else 0,
                1000 * timings.get("eye", 0.0),
                1000 * (timings.get("pose", 0.0) + timings.get("garment", 0.0)),
            )
            self.stats["eye_budget"].append(self.eyes.budget)
            self.stats["plain_covers"].append(int(self.governor.plain))
            g = self.governor
            self.stats["governor"] = [dict(budget_lowered=g.lowered, budget_lowest=g.lowest, plain_periods=g.plain_periods)]
        self.stats["people"].append(decision.n_people)
        self.stats["risky"].append(decision.n_risky)
        return decision

    def _analyse(
        self, frame_id: int, im: np.ndarray, cut: bool, now: float | None, timings: dict[str, float], full: np.ndarray | None
    ) -> Decision:
        cfg, st = self.cfg, self.stats
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        clock = self.clock()
        t = time.perf_counter() if now is None else now  # scene clock: the stream time offline, so runs repeat exactly

        scene_on, scene_from = self._scene_step(frame_id, rgb, im if full is None else full, cut, t)
        st["scene_on"].append(int(scene_on))
        timings["scene"], clock = self._lap(clock)
        if scene_on:  # the whole frame is covered: no person work needed (saves GPU for the stream)
            return Decision(frame_id=frame_id, ok=True, scene=True, scene_from=scene_from)

        people, crops = self.detect(rgb, full, full_bgr=True)
        timings["detect"], clock = self._lap(clock)

        looks = self.eyes.plan(people, self.tracker, cut)
        st["eye_skipped"].append(looks.skipped)
        if looks.look:
            emb = self.models.eye.embed([crops[j] for j in looks.look])
            scores = self.models.eye.prisk(emb)
            if not np.isfinite(scores).all():
                # NaN compares false with every threshold: the person would count as safe. Fail the decision
                raise ValueError("the eye returned a non-finite score")
            for j, e, p in zip(looks.look, emb, scores, strict=True):
                people[j].emb, people[j].p = e, float(p)
        st["embedded"].append(len(looks.look))
        if self.eyes.budget > 0:
            st["eye_put_off"].append(looks.put_off)
            st["eye_early"].append(looks.early)
        waiting = [people[j] for j in looks.waiting]  # not seen and not tracked yet: covered, outside the tracker
        if self.policy.light:  # "light" (the default): nobody is covered before the eye has looked at them
            waiting = []
        people = [p for p in people if p.tid >= -1]  # waiting ids are below -1
        timings["eye"], clock = self._lap(clock)

        self.tracker.assign(people, cut)
        thr = self.thr - SWIM_SHIFT * self.scale * (self.p_swim if cfg.eye.scene_prior and not self.policy.light else 0.0)
        self.policy.update(people, self.tracker, thr, now)
        if self.live.get().everyone:  # the streamer's "dress everyone" switch (ads: every character in the brand)
            for person in people:
                person.risky = True
        self.tracker.forget_old()
        timings["track"], clock = self._lap(clock)
        st["body_rule"].append(self._body_rule(im, rgb, people))
        timings["body"], clock = self._lap(clock)

        risky = [p for p in people if p.risky]
        risky_px = sum(int(p.mask.sum()) for p in risky)
        if cfg.scene.close_area < 1 and risky_px >= cfg.scene.close_area * im.shape[0] * im.shape[1]:
            self.latch.mark_close(t)  # a body close-up: the whole frame is hidden instead of one huge garment
            st["close_cards"].append(1)
            return Decision(frame_id=frame_id, ok=True, scene=True, n_people=len(people), n_risky=len(risky))
        plan = [(p, self._mode(p)) for p in risky]
        dressed = [p for p, mode in plan if mode == "full"]
        poses = dict(zip(map(id, dressed), self.models.pose(rgb, [p.box for p in dressed]), strict=True)) if dressed else {}
        timings["pose"], clock = self._lap(clock)

        safe = [SafePerson(p.mask, p.box, int(p.mask.sum())) for p in people if not p.risky and p.p < SAFE_P * self.scale]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if cfg.render.motion else None
        refs: dict = {}

        def dress(item: tuple[Person, str]) -> Layer | None:
            person, mode = item
            memory = self.tracker.tracks[person.tid].garment
            layer = self.garments.build(im, person, mode, memory, safe, self.tracker.decision, cut, poses.get(id(person)))
            if layer is not None and gray is not None:
                layer.motion = make_motion_ref(gray, person.mask, person.box)
            return layer

        # every person has their own garment memory and nothing else is written: side by side gives the same layers
        built = self._pool.map(dress, plan) if self._pool is not None and len(plan) > 1 else map(dress, plan)
        layers = [layer for layer in built if layer is not None]
        if gray is not None and cfg.render.lookahead:  # a later decision may cover these people here
            covered = {layer.tid for layer in layers}
            for p in people:
                if p.tid not in covered and (ref := make_motion_ref(gray, p.mask, p.box)) is not None:
                    refs[p.tid] = ref
        for layer in layers:
            self.tracker.tracks[layer.tid].last_layer = layer
        for person in waiting:  # a plain cover: nothing is known about them yet, and they have no track to keep it
            layer = self.garments.build(im, person, "small", GarmentMemory(), safe, self.tracker.decision, cut)
            if layer is not None:
                if gray is not None:
                    layer.motion = make_motion_ref(gray, person.mask, person.box)
                layers.append(layer)
        timings["garment"], _ = self._lap(clock)

        people = people + waiting
        risky = risky + waiting
        st["coasted"].append(self._coast(people, layers, cut))

        if cfg.output.diag_out:
            dets = [[p.tid, list(p.box), round(p.p, 3), int(p.risky), int(p.stale)] for p in people]
            self.diag.append([frame_id, dets, round(self.p_swim, 3), round(thr, 4)])  # + beach prior and threshold
        return Decision(
            frame_id=frame_id,
            ok=True,
            layers=layers,
            n_people=len(people),
            n_risky=len(risky),
            people=[(p.tid, p.box) for p in people],
            refs=refs,
        )

    def _lap(self, start: float) -> tuple[float, float]:
        now = self.clock()
        return now - start, now

    def _body_rule(self, im: np.ndarray, rgb: np.ndarray, people: list[Person]) -> int:
        """Women the eye left uncovered: a big back view, a hips close-up or a bare belly is covered anyway
        (analysis/body_rule.py). Only adds covers. Returns how many people it covered."""
        rc = self.cfg.risk
        pmale = getattr(self.models.eye, "pmale", None)
        if not rc.body_rule or pmale is None:
            return 0
        cands = []
        for p in people:
            if p.risky or p.height < rc.body_min_h or p.tid < 0:
                continue
            track = self.tracker.tracks[p.tid]
            emb = track.unit_emb if getattr(track, "unit_emb", None) is not None else p.emb
            if emb is None or float(pmale(np.asarray(emb)[None])[0]) > rc.body_male_max:
                continue
            cands.append(p)
        cands = sorted(cands, key=lambda p: -p.height)[: rc.body_max]
        if not cands:
            return 0
        hits = 0
        for p, (kp, sc) in zip(cands, self.models.pose(rgb, [p.box for p in cands]), strict=True):
            why = body_rule.hit(im, p.mask, p.box, kp, sc)
            if why:
                hits += 1
                p.risky = True
                r = self.tracker.tracks[p.tid].risk
                r.risky, r.safe_run, r.evidence = True, 0, max(r.evidence, S_COVER)
                self.stats["body_" + why].append(1)
        return hits

    def _mode(self, person: Person) -> str:
        """Garment look for a covered person. Someone covered only while their look is put off gets the plain
        cover, and their garment memory stays as it is: nothing says yet that they need a dress."""
        track = self.tracker.tracks[person.tid]
        if person.put_off and not track.risk.risky:
            return "small"
        if self.governor is not None and self.governor.plain:
            return "plain"  # this PC is too slow for dresses right now: the plain cover hides more and costs a tenth
        return choose_mode(person.height, track.garment, self.cfg.garment.small_h)

    def _coast(self, people: list, layers: list, cut: bool) -> int:
        """Covered people the detector missed this time keep their last garment for a few decisions.

        Only where nobody is detected now: if someone is found at that place, the person was seen again (maybe
        under a new identity) and gets a garment of their own; coasting there would draw a double (run 51)."""
        n = self.cfg.risk.coast_decisions
        if n <= 0 or cut:
            return 0
        present = {p.tid for p in people}
        coasted = 0
        for tid, track in self.tracker.tracks.items():
            if tid in present or not track.risk.risky or track.last_layer is None:
                continue
            if track.box[3] - track.box[1] < COAST_MIN_H * ANALYSIS_SIZE[1]:
                continue
            if any(box_iou(track.box, p.box) > COAST_FREE_IOU for p in people):
                continue
            if 0 < self.tracker.decision - track.last_seen <= n:
                layers.append(track.last_layer)
                coasted += 1
        return coasted

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
                if not np.isfinite([self.p_swim, 0.0 if p is None else p]).all():
                    raise ValueError("the scene model returned a non-finite score")  # NaN would read as "no scene"
                self._scene_at = t
                p_explicit = p if cfg.scene.scene_guard else None  # None: no trained head -> HUD only
        hits: list[tuple[int, float]] = []
        if self.hud is not None:
            hits = [(f, ft) for f, ft, kw in self.hud.results() if kw]
            if (cut or t - self._hud_at >= cfg.scene.hud_every_s) and self.hud.submit(frame_id, t, bgr):
                self._hud_at = t
            self.stats["hud_errors"] = [self.hud.errors]
        if not cfg.scene.scene_guard and self.hud is None and cfg.scene.close_area >= 1:
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
