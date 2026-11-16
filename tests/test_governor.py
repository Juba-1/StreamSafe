"""What the engine does on a PC that turns out too slow in mid-stream (analysis/headroom.py, BudgetGovernor): fewer
looks per decision, then plain covers instead of dresses. Both only ever add cover. Plus the pieces that make a
decision cheaper without changing a pixel: garments built side by side, and the robe opened inside its own box."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from streamsafe.analysis.headroom import CALM_DECISIONS, FULL_BATCH, GOVERNOR_FLOOR, PLAIN_DECISIONS, PLAIN_MAX, BudgetGovernor
from streamsafe.config import Config
from streamsafe.garment import shapes
from streamsafe.runtime.report import slate_episodes, summarize
from tests.stubs import RED, stub_analyzer
from tests.test_eye_budget import MIXED, crowd
from tests.test_headroom import SLOW, timed_analyzer

LIMIT = 180.0


# ------------------------------------------------------------------------------------------- the rule itself
def test_a_slow_decision_lowers_the_budget_at_once_in_proportion_and_never_below_one():
    g = BudgetGovernor(7, LIMIT)
    assert g.update(170.0, 7) == 7, "under the limit: nothing changes"
    assert g.update(190.0, 7) == 6, "a little over: one look fewer"
    assert g.update(400.0, 6) == 2, "far over: as many looks as would have fitted (6 x 180 / 400)"
    assert g.update(900.0, 2) == GOVERNOR_FLOOR and g.update(900.0, 1) == GOVERNOR_FLOOR
    assert g.lowered == 3
    quiet = BudgetGovernor(7, LIMIT)
    assert quiet.update(500.0, 0) == 7, "slow without a single look: the eye was not the reason"
    assert quiet.update(200.0, 3) == 2, "slow with fewer looks than the budget: fewer than that next time"


def test_no_budget_at_start_gets_one_when_the_pc_turns_out_slow_and_loses_it_again_when_calm():
    g = BudgetGovernor(0, LIMIT)
    assert g.update(100.0, 12) == 0
    assert g.update(360.0, 12) == 6
    for k in range(6, FULL_BATCH):
        for _ in range(CALM_DECISIONS - 1):
            assert g.update(60.0, k) == k
        assert g.update(60.0, k) == (k + 1 if k + 1 < FULL_BATCH else 0)
    assert g.budget == 0, "back to no limit, exactly as it started"


def test_the_budget_is_raised_slowly_only_on_evidence_and_never_above_its_start():
    g = BudgetGovernor(5, LIMIT)
    g.update(400.0, 5)
    assert g.budget == 2
    for _ in range(10 * CALM_DECISIONS):
        g.update(20.0, 1)  # quiet scenes: one look, fast. They say nothing about three looks
    assert g.budget == 2
    for _ in range(CALM_DECISIONS - 1):
        g.update(90.0, 2)
    g.update(150.0, 2)  # 150 ms with two looks: no room for a third (225 ms). The count starts again
    for _ in range(CALM_DECISIONS - 1):
        g.update(90.0, 2)
    assert g.budget == 2
    g.update(90.0, 2)
    assert g.budget == 3
    for _ in range(100 * CALM_DECISIONS):
        g.update(30.0, g.budget)
    assert g.budget == 5, "never above where it started"
    rng = np.random.default_rng(1)
    a, b = BudgetGovernor(6, LIMIT), BudgetGovernor(6, LIMIT)
    for _ in range(2000):
        ms, crops, eye, dress = rng.uniform(20, 400), int(rng.integers(0, 9)), rng.uniform(0, 200), rng.uniform(0, 200)
        assert a.update(ms, crops, eye, dress) == b.update(ms, crops, eye, dress), "the same times give the same budget"
        assert GOVERNOR_FLOOR <= a.budget <= 6 and a.plain == b.plain


def test_slow_dresses_switch_to_plain_covers_for_a_while_and_longer_when_it_happens_again():
    g = BudgetGovernor(0, LIMIT)
    g.update(250.0, 4, eye_ms=150.0, dress_ms=60.0)
    assert not g.plain, "the eye was the cost: that is the budget's job"
    g.update(250.0, 1, eye_ms=30.0, dress_ms=160.0)
    assert g.plain and g.plain_periods == 1
    for _ in range(PLAIN_DECISIONS - 1):
        g.update(40.0, 1, 30.0, 5.0)
        assert g.plain
    g.update(40.0, 1, 30.0, 5.0)
    assert not g.plain, "dresses come back after the period"
    g.update(250.0, 1, eye_ms=30.0, dress_ms=160.0)  # ... and are too slow again at once
    assert g.plain
    for _ in range(2 * PLAIN_DECISIONS - 1):
        g.update(40.0, 1, 30.0, 5.0)
    assert g.plain, "the second period is twice as long"
    g.update(40.0, 1, 30.0, 5.0)
    assert not g.plain
    for _ in range(20):  # a PC that can never afford them: the periods grow to the cap and stay there
        g.update(250.0, 1, 30.0, 160.0)
        while g.plain:
            g.update(40.0, 1, 30.0, 5.0)
    assert g._plain_period == PLAIN_MAX
    for _ in range(PLAIN_DECISIONS + 1):
        g.update(40.0, 1, 30.0, 5.0)
    g.update(250.0, 1, 30.0, 160.0)
    assert g._plain_period == PLAIN_DECISIONS, "after a long good stretch the next plain period is short again"


# ------------------------------------------------------------------------------------- through the analyzer
def test_a_gpu_taken_away_in_mid_stream_costs_looks_not_cover():
    risky = {i for i, c in enumerate(MIXED) if c == RED}
    an, clock, sizes = timed_analyzer(**SLOW)  # realtime, defaults: automatic budget and governor
    an.warmup(clock)
    assert an.eyes.budget == 7 and an.governor is not None
    slow_eye = an.models.eye.embed

    def loaded(crops):  # a game starts: every eye run takes three times as long from now on
        clock.ms += 2 * (26.0 + 18.5 * (len(crops) - 1))
        return slow_eye(crops)

    an.models.eye.embed = loaded
    del sizes[:]
    for f in range(40):
        d = an(f * 3, crowd(MIXED), cut=f == 0, now=f / 10)
        assert d.ok, an.stats["analyzer_errors"]
        drawn = {layer.tid for layer in d.layers}
        covered = {round((box[0] + box[2] - 120) / 190) for tid, box in d.people if tid in drawn}
        assert risky <= covered, "a risky person is covered in every decision, whatever the budget does"
    assert sizes[0] == 7 and max(sizes[1:]) <= 2, "one slow decision, then the eye gets what fits"
    assert max(an.stats["ms_total"][1:]) <= LIMIT, "and no later decision comes near the slate limit"
    assert an.stats["governor"][0]["budget_lowered"] >= 1 and min(an.stats["eye_budget"]) >= GOVERNOR_FLOOR


def test_offline_runs_and_a_switched_off_governor_never_follow_the_clock():
    for settings in (dict(mode="offline"), dict(eye_budget_adapt=False)):
        an, clock, _ = timed_analyzer(**SLOW, **settings)
        an.warmup(clock)
        assert an.governor is None
        an(0, crowd(MIXED), cut=True, now=0.0)
        assert "eye_budget" not in an.stats and "plain_covers" not in an.stats


def _layers(plain: bool, threads: int = 1):
    cfg = Config.from_env(env={}, mode="realtime", garment_threads=threads)
    an = stub_analyzer(cfg, 2.0)
    an.warmup()
    assert an.governor is not None
    an.governor.limit_ms = 1e9  # the test machine's own speed must not matter
    an.governor.plain = plain
    an.governor._plain_left = 10**6
    im = crowd([RED, RED, RED, RED], heights=[300, 260, 200, 110])
    out = []
    for f in range(3):
        d = an(f * 3, im, cut=f == 0, now=f / 10)
        assert d.ok, an.stats["analyzer_errors"]
        out.append(d)
    return an, im, out


def test_plain_covers_hide_the_whole_person():
    an, im, decisions = _layers(plain=True)
    _, _, dressed = _layers(plain=False)
    det = an.models.detector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    for d, ref in zip(decisions, dressed, strict=True):
        assert len(d.layers) == 4 == len(ref.layers)
        canvas = np.zeros((1080, 1920), bool)
        for layer in d.layers:
            h, w = layer.shape
            canvas[layer.y : layer.y + h, layer.x : layer.x + w] |= layer.mask
            assert len(np.unique(layer.fabric.reshape(-1, 3), axis=0)) == 1, "plain cloth, no shading work"
        for _box, mask in det:
            big = cv2.resize(mask.astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
            assert (canvas & big).sum() >= 0.999 * big.sum(), "every pixel of every person is under the cover"
    assert an.stats["plain_covers"] == [1, 1, 1]


@pytest.mark.parametrize("plain", [False, True])
def test_garments_built_side_by_side_are_the_same_garments(plain):
    _, _, one = _layers(plain, threads=1)
    an, _, many = _layers(plain, threads=3)
    assert an._pool is not None
    for a, b in zip(one, many, strict=True):
        assert [layer.tid for layer in a.layers] == [layer.tid for layer in b.layers]
        for x, y in zip(a.layers, b.layers, strict=True):
            assert (x.y, x.x, x.centroid) == (y.y, y.x, y.centroid)
            assert np.array_equal(x.mask, y.mask) and np.array_equal(x.fabric, y.fabric)


# ------------------------------------------------------------------------------ the robe, opened in its own box
def _robe_whole_canvas(mask: np.ndarray) -> np.ndarray:
    """``shapes.robe_region`` as it was before: the opening runs over the whole upright canvas."""
    real = cv2.morphologyEx
    boxes = []

    def spy(src, op, kernel, *a, **k):
        boxes.append(src.shape)
        return real(src, op, kernel, *a, **k)

    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, evec = np.linalg.eigh(np.cov((pts - c).T))
    axis = evec[:, 1] if evec[1, 1] >= 0 else -evec[:, 1]
    angle = np.degrees(np.arctan2(axis[0], axis[1]))
    h, w = mask.shape
    rot = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), -angle, 1.0)
    big = int(np.hypot(h, w))
    rot[0, 2] += (big - w) / 2
    rot[1, 2] += (big - h) / 2
    upright = cv2.warpAffine(mask.astype(np.uint8), rot, (big, big), flags=cv2.INTER_NEAREST)
    rows = np.where(upright.any(1))[0]
    top, bot = rows.min(), rows.max()
    left, right = np.full(big, -1.0), np.full(big, -1.0)
    for y in rows:
        cols = np.where(upright[y])[0]
        left[y], right[y] = cols.min(), cols.max()
    height = max(1, bot - top)
    idx = np.arange(top, bot + 1)
    lv, rv = left[idx], right[idx]
    ok = lv >= 0
    lv, rv = np.interp(idx, idx[ok], lv[ok]), np.interp(idx, idx[ok], rv[ok])
    k = max(3, int(height * 0.08)) | 1
    lv = cv2.GaussianBlur(lv.reshape(-1, 1), (1, k), 0).ravel()
    rv = cv2.GaussianBlur(rv.reshape(-1, 1), (1, k), 0).ravel()
    t = (idx - top) / height
    flare = (0.06 + 0.22 * np.clip((t - 0.2) / 0.8, 0, 1)) * height * 0.35
    extra = np.maximum(0, (np.where(t > 0.18, 0.32 * height, 0) - (rv - lv)) / 2)
    poly = np.concatenate([np.stack([lv - flare - extra, idx], 1), np.stack([rv + flare + extra, idx], 1)[::-1]])
    robe = np.zeros_like(upright)
    cv2.fillPoly(robe, [poly.astype(np.int32)], 1)
    ks = max(3, int(0.10 * height)) | 1
    robe = cv2.morphologyEx(robe, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    return cv2.warpAffine(robe, cv2.invertAffineTransform(rot), (w, h), flags=cv2.INTER_NEAREST).astype(bool) | mask


def test_the_robe_opened_in_its_own_box_is_the_same_robe():
    rng = np.random.default_rng(3)
    for case in range(60):
        h, w = int(rng.integers(60, 420)), int(rng.integers(40, 300))
        mask = np.zeros((h, w), np.uint8)
        centre = (int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.1, 0.9) * h))  # often cut by the window's edge
        axes = (max(4, int(rng.uniform(0.08, 0.35) * w)), max(8, int(rng.uniform(0.2, 0.6) * h)))
        cv2.ellipse(mask, centre, axes, float(rng.uniform(-80, 80)), 0, 360, 1, -1)
        if case % 3 == 0:  # an arm sticking out
            cv2.line(mask, centre, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), 1, max(2, w // 20))
        mask = mask.astype(bool)
        if mask.sum() < 50:
            continue
        region, info = shapes.robe_region(mask)
        assert info is not None and np.array_equal(region, _robe_whole_canvas(mask)), case


# --------------------------------------------------------------------------------------------- the run report
def test_the_run_summary_counts_slate_episodes():
    assert slate_episodes([]) == []
    assert slate_episodes([5, 6, 7, 40, 41, 90]) == [(5, 7), (40, 41), (90, 90)]
    frames = [*range(100, 130), *range(400, 403), 900]
    s = summarize({"failclosed_frames": frames}, {}, 1000, 300, 30.0, {})["summary"]
    assert (s["failclosed_frames"], s["slate_episodes"], s["slate_longest_frames"]) == (34, 3, 30)
    quiet = summarize({}, {}, 1000, 300, 30.0, {})["summary"]
    assert (quiet["slate_episodes"], quiet["slate_longest_frames"]) == (0, 0)
