"""Component-level rules: fail-closed gate, decision selection, config, risk policy, tracker."""

from __future__ import annotations

import numpy as np
import pytest

from streamsafe.analysis.risk import RiskPolicy
from streamsafe.analysis.tracks import Tracker
from streamsafe.config import Config, RenderConfig, RiskConfig
from streamsafe.render.compositor import usable_decisions
from streamsafe.runtime.cuts import CutDetector
from streamsafe.runtime.gate import OutputGate
from streamsafe.types import Decision, Layer, Person

W, H = 1920, 1080


def frame(value: int = 100) -> np.ndarray:
    return np.full((H, W, 3), value, np.uint8)


def square_layer(tid: int = 0) -> Layer:
    mask = np.ones((100, 100), bool)
    return Layer(tid=tid, y=500, x=900, mask=mask, fabric=np.full((100, 100, 3), 7, np.uint8), centroid=(550.0, 950.0))


# ---------------------------------------------------------------------------------- output gate
def test_gate_without_decisions_shows_the_slate():
    gate = OutputGate(RenderConfig(), (W, H))
    out = gate.render(0, frame())
    assert out is gate.slate
    assert gate.stats["failclosed_frames"] == [0]


def test_gate_failed_decision_fails_closed_after_hold():
    gate = OutputGate(RenderConfig(hold=2, motion=False), (W, H))
    gate.add_decision(Decision(frame_id=0, ok=True, layers=[square_layer()]))
    first = gate.render(0, frame())
    assert (first[550, 950] == 7).all(), "garment drawn"
    gate.add_decision(Decision(frame_id=1, ok=False))
    gate.add_decision(Decision(frame_id=2, ok=False))
    gate.add_decision(Decision(frame_id=3, ok=False))
    assert gate.render(1, frame()) is first, "1st bad frame repeats the last protected frame"
    assert gate.render(2, frame()) is first, "2nd bad frame repeats it too"
    assert gate.render(3, frame()) is gate.slate, "then the slate"
    assert gate.stats["held_frames"] == [1, 2]
    assert gate.stats["failclosed_frames"] == [3]


def test_gate_never_uses_a_decision_from_another_shot():
    gate = OutputGate(RenderConfig(motion=False), (W, H))
    gate.add_decision(Decision(frame_id=0, ok=True, layers=[square_layer()]))
    gate.mark_cut(5, True)
    gate.render(4, frame())
    assert gate.render(5, frame()) is not gate.slate  # held once (hold=2)...
    gate.render(6, frame())
    assert gate.render(7, frame()) is gate.slate  # ...then fail closed: frame 0's garment is from another shot


def test_usable_decisions_window_and_cuts():
    keys = [0, 3, 6, 30]
    no_cut = lambda a, b: False  # noqa: E731
    assert usable_decisions(4, keys, no_cut, 18) == [3, 6]
    assert usable_decisions(6, keys, no_cut, 18) == [6]  # 30 is 24 frames away
    assert usable_decisions(14, keys, no_cut, 18) == [6, 30]
    assert usable_decisions(10, keys, no_cut, 18) == [6]  # 30 is 20 frames away
    assert usable_decisions(4, keys, lambda a, b: a < 5 <= b, 18) == [3]  # cut at 5 between 4 and 6
    assert usable_decisions(50, keys, no_cut, 18) == []


# ------------------------------------------------------------------------------------- config
def test_config_env_and_overrides():
    cfg = Config.from_env(env={"DELAY": "1.5", "EYE_EVERY": "1", "MOTION": "false"}, color="black")
    assert cfg.runtime.delay == 1.5
    assert cfg.eye.eye_every == 1
    assert cfg.render.motion is False
    assert cfg.garment.color == "black"
    with pytest.raises(KeyError):
        Config.from_env(env={}, no_such_setting=1)


# ---------------------------------------------------------------------------------- cut detector
def test_cut_detector():
    cd = CutDetector()
    a, b = np.full((540, 960, 3), (90, 120, 140), np.uint8), np.full((540, 960, 3), (150, 90, 60), np.uint8)
    assert cd(a) is False
    assert cd(a) is False
    assert bool(cd(b)) is True


# ------------------------------------------------------------------------------ tracker / risk
def _person(x: int, p: float, emb: np.ndarray) -> Person:
    mask = np.zeros((540, 960), bool)
    mask[100:300, x : x + 60] = True
    return Person(box=(x, 100, x + 60, 300), mask=mask, emb=emb, p=p)


def test_tracker_keeps_identity_and_skips_the_eye_for_a_still_person():
    emb = np.zeros(8, np.float32)
    emb[0] = 1
    tr = Tracker()
    policy = RiskPolicy(RiskConfig(), 0.1746, lambda e: np.full(len(e), 0.95))
    for step in range(4):
        people = [_person(100 + step, 0.95, emb)]
        hinted = tr.motion_hints(people, cut=False, eye_every=3)
        tr.assign(people, cut=False)
        policy.update(people, tr, 0.1746, now=step / 10)
        assert people[0].tid == 0, "same person, same track"
        assert people[0].risky
        if step == 2:
            assert hinted == 1 and people[0].stale, "eye skipped while the person barely moved"


def test_risk_release_needs_time_for_borderline_readings():
    """0.06 is below the threshold but not 'very sure' (fast clear needs < 0.08 * scale = 0.035):
    the cover stays on until the evidence is clearly safe AND release_s seconds have passed."""
    emb = np.zeros(8, np.float32)
    emb[0] = 1
    reading = {"p": 0.95}
    tr = Tracker()
    policy = RiskPolicy(RiskConfig(release_s=1.0), 0.1746, lambda e: np.full(len(e), reading["p"]))

    def step(t: float) -> bool:
        people = [_person(100, reading["p"], emb)]
        tr.assign(people, cut=False)
        policy.update(people, tr, 0.1746, now=t)
        return people[0].risky

    times = [i / 10 for i in range(80)]
    assert all(step(t) for t in times[:5])
    reading["p"] = 0.06
    released = next(t for t in times[5:] if not step(t))
    assert released - times[5] >= 1.0


def _run(policy, tr, emb, p, steps, start=0, x=100):
    out = []
    for i in range(start, start + steps):
        people = [_person(x, p, emb)]
        tr.assign(people, cut=False)
        policy.update(people, tr, 0.1746, now=i / 10)
        out.append(people[0].risky)
    return out


def test_risk_fast_clear_only_for_marginal_covers():
    """Two very low readings release a person who was only marginally risky, never one who read clearly
    risky (a bent-over or half-off-screen pose reads low: the golden rule keeps the cover)."""
    emb = np.zeros(8, np.float32)
    emb[0] = 1
    tr = Tracker()
    reading = {"p": 0.25}  # above thr 0.1746, below 1.5 x thr
    policy = RiskPolicy(RiskConfig(), 0.1746, lambda e: np.full(len(e), reading["p"]))
    assert _run(policy, tr, emb, 0.25, 12)[-1]
    reading["p"] = 0.01
    assert _run(policy, tr, emb, 0.01, 3, start=12) == [True, False, False], "marginal cover: fast clear"

    tr2 = Tracker()
    reading["p"] = 0.5
    assert _run(policy, tr2, emb, 0.5, 5)[-1]
    reading["p"] = 0.01
    assert all(_run(policy, tr2, emb, 0.01, 3, start=5)), "clearly risky: no fast clear"


def test_risk_handoff_keeps_the_cover_when_identity_breaks():
    """A covered person whose track is lost (new pose, edge of frame) and re-found as a new track at the same
    place stays covered at once, instead of facing the strict first-look rule for big people."""
    tr = Tracker()
    reading = {"p": 0.5}
    policy = RiskPolicy(RiskConfig(), 0.1746, lambda e: np.full(len(e), reading["p"]))
    a = np.zeros(8, np.float32)
    a[0] = 1
    assert _run(policy, tr, a, 0.5, 5)[-1]
    b = np.zeros(8, np.float32)
    b[1] = 1  # looks different now: the tracker makes a new track
    reading["p"] = 0.2  # reads low-ish in the new pose (below the first-look bar 1.6 x thr)
    people = [_person(104, 0.2, b)]
    tr.assign(people, cut=False)
    policy.update(people, tr, 0.1746, now=0.5)
    assert people[0].tid != 0 and people[0].risky, "new track took over the cover"

    tr2 = Tracker()
    policy2 = RiskPolicy(RiskConfig(handoff=False), 0.1746, lambda e: np.full(len(e), reading["p"]))
    reading["p"] = 0.5
    _run(policy2, tr2, a, 0.5, 5)
    reading["p"] = 0.2
    people = [_person(104, 0.2, b)]
    tr2.assign(people, cut=False)
    policy2.update(people, tr2, 0.1746, now=0.5)
    assert not people[0].risky, "without handoff the identity break uncovers her (the run 50 bug)"


def test_brand_slate_uses_the_streamers_image_and_colour(tmp_path):
    """The slate / scene card can carry the streamer's logo (exact colours) on its own background colour."""
    import cv2

    from streamsafe.config import BrandConfig
    from streamsafe.garment.fabric import parse_color
    from streamsafe.render.slate import brand_frames

    logo = np.zeros((100, 100, 3), np.uint8)
    logo[:] = (30, 30, 30)
    logo[30:70, 30:70] = (0, 0, 255)  # red square in the middle
    path = str(tmp_path / "logo.png")
    cv2.imwrite(path, logo)
    slate, card = brand_frames(640, 360, BrandConfig(slate_image=path, slate_text="", scene_text=""))
    assert tuple(slate[2, 2]) == (30, 30, 30), "background = the image's border colour"
    assert tuple(slate[180, 320]) == (0, 0, 255), "logo centre in its exact colour"
    assert (card == slate).all(), "scene card uses the slate image unless scene_image is set"
    slate, _ = brand_frames(640, 360, BrandConfig(slate_color="#102030"))
    assert tuple(slate[2, 2]) == parse_color("#102030") == (0x30, 0x20, 0x10)


def test_track_smoother_steadies_wobble_follows_motion_and_resets_on_jumps():
    from streamsafe.render.smooth import TrackSmoother

    rng = np.random.default_rng(0)
    sm = TrackSmoother(alpha=0.5, beta=0.15, reset_frac=0.25)
    truth = [(100 + 3.0 * v, 200 + 2.0 * v) for v in range(60)]  # steady walk
    meas = [(y + rng.normal(0, 2), x + rng.normal(0, 2)) for y, x in truth]
    out = [sm.place(1, v, 0, y, x, 300) for v, (y, x) in enumerate(meas)]

    def shake(seq):
        return float(np.mean([abs(seq[i + 1][0] - 2 * seq[i][0] + seq[i - 1][0]) for i in range(1, len(seq) - 1)]))

    assert shake(out) < 0.6 * shake(meas), "wobble reduced"
    err = np.mean([abs(o[0] - t[0]) for o, t in zip(out[20:], truth[20:], strict=True)])
    assert err < 2.0, "follows a steady walk without trailing"
    y, _ = sm.place(1, 60, 0, 900.0, 200.0, 300)  # cut / re-detection far away
    assert y == 900.0, "a jump resets at once (never drifts off the person)"
    assert sm.place(2, 5, 0, 10.0, 10.0, 100) == (10.0, 10.0), "a new person starts at its measurement"


def test_fast_mover_gets_a_margin_slow_mover_does_not():
    from streamsafe.analysis.tracks import GarmentMemory
    from streamsafe.config import GarmentConfig
    from streamsafe.garment.builder import GarmentBuilder

    b = GarmentBuilder(GarmentConfig(), 2.0)
    cover = np.zeros((200, 200), bool)
    cover[50:150, 80:120] = True
    keep_off = np.zeros_like(cover)

    class P:  # person box now: x 80-120, y 50-150 (h 100)
        box = (80, 50, 120, 150)

    mem = GarmentMemory()
    mem.cover, mem.cover_at = (cover, (79, 50, 119, 150), (0, 0)), 4  # moved 1 px: slow
    assert (b._motion_margin(cover, keep_off, P, mem, 5, False) == cover).all()
    mem.cover = (cover, (60, 50, 100, 150), (0, 0))  # moved 20 px = 0.2 x height: fast
    out = b._motion_margin(cover, keep_off, P, mem, 5, False)
    assert out[100, 135] and out[100, 65], "covers where she runs to and comes from"
    assert out.sum() > cover.sum()
    assert (b._motion_margin(cover, keep_off, P, mem, 5, True) == cover).all(), "never across a cut"


def test_damaged_head_files_are_refused_at_start(tmp_path):
    from streamsafe.models.heads import FAST_FILE, MALE_FILE, THR_FILE, RiskHeads

    (tmp_path / FAST_FILE).write_text("100;0.5;10,20,30")
    (tmp_path / MALE_FILE).write_text("100;-0.5;1,2,3")
    (tmp_path / THR_FILE).write_text("0.2104")
    assert RiskHeads.load(tmp_path).thr == pytest.approx(0.2104)
    (tmp_path / MALE_FILE).write_text("100;-0.5;1,2")  # size mismatch
    with pytest.raises(ValueError):
        RiskHeads.load(tmp_path)
    (tmp_path / MALE_FILE).write_text("100;-0.5;1,2,3")
    (tmp_path / THR_FILE).write_text("0")  # would cover nobody / everybody
    with pytest.raises(ValueError):
        RiskHeads.load(tmp_path)


def test_covered_person_missed_by_the_detector_keeps_the_garment_briefly():
    """Run 50 club dancer: detection dropped her for 0.4 s and the cover vanished. A covered person the
    detector misses keeps the last garment for coast_decisions, then (still missing) it is dropped."""
    import cv2

    from tests.stubs import RED, stub_analyzer

    cfg = Config.from_env(env={}, mode="offline")
    an = stub_analyzer(cfg, 1.0)

    def frame(visible: bool) -> np.ndarray:
        im = np.full((540, 960, 3), 90, np.uint8)
        cv2.ellipse(im, (480, 300), (40, 110), 0, 0, 360, RED if visible else (90, 90, 90), -1)
        return im

    for f in range(6):
        d = an(f * 3, frame(True), cut=False, now=f / 10)
    assert d.layers, "covered while visible"
    tid = d.layers[0].tid
    kept = [bool([la for la in an(f * 3, frame(False), cut=False, now=f / 10).layers if la.tid == tid]) for f in range(6, 14)]
    n = cfg.risk.coast_decisions
    assert kept[:n] == [True] * n, "coasts while the detector misses her"
    assert not any(kept[n + 1 :]), "then dropped: no endless ghost"
    assert not an(99, frame(False), cut=True, now=2.0).layers, "never across a cut"
