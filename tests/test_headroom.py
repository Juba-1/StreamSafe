"""Realtime headroom (analysis/headroom.py): the model times measured at start, the rate check, and the automatic
eye budget. The rule is checked on the numbers of run 57 (RTX 3090) and through the analyzer with a fake clock."""

from __future__ import annotations

import numpy as np
import pytest

from streamsafe.analysis import headroom
from streamsafe.analysis.analyzer import Analyzer
from streamsafe.analysis.headroom import FULL_BATCH, MIN_BUDGET, Headroom
from streamsafe.config import Config
from streamsafe.models import Models
from tests.stubs import RED, BlobDetector, ColourEye, MagentaScene, NoPose
from tests.test_eye_budget import MIXED, crowd

STEP, LIMIT = 100.0, 180.0  # 10 decisions/s at 30 fps; 0.6 of the 9-frame slate limit


def path(fixed: float, eye_one: float, eye_full: float) -> Headroom:
    return Headroom(fixed, eye_one, eye_full, STEP, LIMIT)


# run 57, RTX 3090: detector 18 ms + pose 5 ms on TensorRT, one eye crop 8.3 ms alone and 5.2 ms in a full batch
TRT_3090 = path(18 + 5 + headroom.CPU_MS, 8.3, 16 * 5.2)
# the same card on CUDA fp32: one eye crop 26 ms alone, 19 ms in a full batch (detector and pose as in run 55)
CUDA_3090 = path(20 + 7 + headroom.CPU_MS, 26.0, 16 * 19.0)


def test_a_fast_tensorrt_path_needs_no_budget_and_holds_the_rate():
    assert TRT_3090.budget == 0 and TRT_3090.holds_rate and TRT_3090.warning() == ""
    assert TRT_3090.decision_ms(FULL_BATCH) < 0.7 * LIMIT, "not a close call: the measured path has room to spare"
    assert TRT_3090.decisions_per_s == 10.0, "never reported above the target rate"


def test_the_cuda_fallback_gets_a_budget_and_is_named_as_too_slow():
    assert CUDA_3090.budget == 7
    assert CUDA_3090.decision_ms(7) <= LIMIT < CUDA_3090.decision_ms(8), "the largest batch that keeps a decision off the slate"
    assert not CUDA_3090.holds_rate and 8.0 < CUDA_3090.decisions_per_s < 8.6
    text = CUDA_3090.warning()
    assert "8.3 decisions/s" in text and "target 10" in text and "121 ms" in text
    report = CUDA_3090.as_dict()
    assert report["budget"] == 7 and report["holds_rate"] is False and report["eye_full_ms"] == 304.0 and report["limit_ms"] == LIMIT


def test_the_budget_is_bounded_and_never_grows_on_a_slower_path():
    hopeless = path(400.0, 90.0, 1400.0)  # not even one crop fits: the floor, and the warning says the rest
    assert hopeless.budget == MIN_BUDGET and not hopeless.holds_rate and hopeless.decisions_per_s < 2.5
    assert path(30.0, 10.0, 149.9).budget == 0 and path(30.0, 10.0, 150.1).budget == FULL_BATCH - 1, "the edge is exact"
    rng = np.random.default_rng(0)
    for _ in range(500):
        fixed, one = rng.uniform(5, 300), rng.uniform(1, 80)
        full = one * rng.uniform(4, 16)
        h, slower = path(fixed, one, full), path(fixed * 1.3, one * 1.3, full * 1.3)
        assert h.budget == 0 or MIN_BUDGET <= h.budget < FULL_BATCH
        assert h.budget == path(fixed, one, full).budget, "the same numbers give the same budget"
        if h.budget:
            assert slower.budget and slower.budget <= h.budget, "slower models never get to look at more people"
            assert h.budget == MIN_BUDGET or h.decision_ms(h.budget) <= LIMIT
        assert slower.holds_rate <= h.holds_rate


# ----------------------------------------------------------------------- through the analyzer, with a fake clock
class Clock:
    def __init__(self) -> None:
        self.ms = 0.0

    def __call__(self) -> float:
        return self.ms / 1000.0


def timed_analyzer(det: float, pose: float, eye_one: float, eye_each: float, **settings) -> tuple[Analyzer, Clock, list[int]]:
    """Stand-in models that take a fixed time on a clock of their own: an eye run costs
    ``eye_one + eye_each * (crops - 1)`` ms. Returns the eye batch sizes as they happen."""
    clock, sizes = Clock(), []

    class Detector(BlobDetector):
        def __call__(self, rgb):
            clock.ms += det
            return super().__call__(rgb)

    class Pose(NoPose):
        def __call__(self, rgb, boxes):
            clock.ms += pose
            return super().__call__(rgb, boxes)

    class Eye(ColourEye):
        def embed(self, crops):
            clock.ms += eye_one + eye_each * (len(crops) - 1)
            sizes.append(len(crops))
            return super().embed(crops)

    cfg = Config.from_env(env={}, **{"mode": "realtime", **settings})
    return Analyzer(cfg, Models(detector=Detector(), eye=Eye(), pose=Pose()), 1.0), clock, sizes


SLOW = dict(det=20.0, pose=7.0, eye_one=26.0, eye_each=(304.0 - 26.0) / 15)  # the CUDA numbers above
FAST = dict(det=18.0, pose=5.0, eye_one=8.3, eye_each=(83.2 - 8.3) / 15)


def test_warmup_measures_this_pcs_models():
    an, clock, sizes = timed_analyzer(**SLOW)
    an.warmup(clock)
    h = an.headroom
    assert h is not None and (h.step_ms, h.limit_ms) == (STEP, LIMIT)
    assert h.fixed_ms == pytest.approx(39.0) and h.eye_one_ms == pytest.approx(26.0) and h.eye_full_ms == pytest.approx(304.0)
    assert sizes[-2 * headroom.RUNS :] == [1] * headroom.RUNS + [FULL_BATCH] * headroom.RUNS, "timed after every size was warmed"
    assert an.stats["headroom"] == [dict(h.as_dict(), budget_applied=7)] and h.budget == 7


def test_the_automatic_budget_is_on_by_default_realtime_only_and_an_explicit_budget_wins():
    for settings, expect in (
        ({}, 7),  # the default since the laptop field report: a slow path gets its budget without being asked
        (dict(eye_budget_auto=False), 0),  # switched off: measured and reported, nothing changed
        (dict(mode="offline"), 0),  # benchmarks must not depend on the machine
        (dict(eye_budget_auto=True, mode="offline"), 0),
        (dict(eye_budget_auto=True, eye_budget=4), 4),
    ):
        an, clock, _ = timed_analyzer(**SLOW, **settings)
        an.warmup(clock)
        assert an.eyes.budget == expect, settings
        assert an.stats["headroom"][0].get("budget_applied") == (7 if expect == 7 else None)
    fast, clock, _ = timed_analyzer(**FAST, eye_budget_auto=True)
    fast.warmup(clock)
    assert fast.eyes.budget == 0 and fast.stats["headroom"][0]["budget_applied"] == 0, "a fast TensorRT path stays without a budget"


def test_with_the_automatic_budget_the_slow_path_never_exceeds_it_and_the_risky_stay_covered():
    risky = {i for i, c in enumerate(MIXED) if c == RED}
    results = {}
    for name, times in (("slow", SLOW), ("fast", FAST)):
        an, clock, sizes = timed_analyzer(**times, eye_budget_auto=True)
        an.warmup(clock)
        del sizes[:]
        covered = []
        for f in range(12):
            d = an(f * 3, crowd(MIXED), cut=f == 0, now=f / 10)
            assert d.ok, an.stats["analyzer_errors"]
            drawn = {layer.tid for layer in d.layers}
            covered.append({round((box[0] + box[2] - 120) / 190) for tid, box in d.people if tid in drawn})
        results[name] = (list(sizes), covered)
        assert all(risky <= c for c in covered), "a risky person is covered in every decision"
    slow_sizes, slow_cover = results["slow"]
    assert max(slow_sizes) == 7 and slow_cover[0] == risky | {7, 8}, "nine new people, seven looks: the two still waiting are covered"
    fast_sizes, fast_cover = results["fast"]
    assert max(fast_sizes) == 9 and all(c == risky for c in fast_cover), "the fast path decides exactly as without a budget"


def test_the_run_report_carries_the_measurement(tmp_path, stub_run):
    from tests.stubs import make_video

    path_ = str(tmp_path / "walk.mp4")
    make_video(path_, n=30)
    report, _ = stub_run(path_, "offline")
    s = report["summary"]
    assert s["headroom"]["budget"] == 0 and s["headroom"]["holds_rate"] is True and s["headroom"]["step_ms"] == 100.0
    assert s["device_notes"] == [] and s["gpu"] and "headroom" not in report and "device_notes" not in report


# ------------------------------------------------------------------- a model that answers NaN fails closed
def test_a_nan_from_the_eye_or_the_scene_model_fails_the_decision():
    class NanEye(ColourEye):
        def prisk(self, emb):
            return np.full(len(emb), np.nan)

    class NanScene(MagentaScene):
        def __call__(self, rgb):
            return 0.0, float("nan")

    cfg = Config.from_env(env={}, mode="offline")
    for models, what in (
        (Models(detector=BlobDetector(), eye=NanEye(), pose=NoPose()), "eye"),
        (Models(detector=BlobDetector(), eye=ColourEye(), pose=NoPose(), scene=NanScene()), "scene"),
    ):
        an = Analyzer(cfg, models, 1.0)
        d = an(0, crowd(MIXED), cut=True, now=0.0)
        assert not d.ok and not d.layers, "NaN compares false with every threshold: nobody would be covered"
        assert "non-finite" in an.stats["analyzer_errors"][0] and what in an.stats["analyzer_errors"][0]
