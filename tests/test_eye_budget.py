"""The eye budget (analysis/eye_schedule.py): at most EYE_BUDGET crops per decision, and whoever has to wait for
the eye is covered until it has seen them. Stand-in models, still people side by side on a grey frame."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from streamsafe.analysis.risk import RiskPolicy
from streamsafe.analysis.tracks import Tracker
from streamsafe.config import Config, RiskConfig
from streamsafe.types import Person
from tests.stubs import BLUE, GREEN, RED, make_video, stub_analyzer
from tests.test_engine import changed, source_frames

THR = 0.1746
MIXED = [RED, GREEN, GREEN, RED, GREEN, GREEN, RED, GREEN, GREEN]


def crowd(colours: list, heights: list[int] | None = None) -> np.ndarray:
    im = np.full((540, 960, 3), 90, np.uint8)
    for i, col in enumerate(colours):
        cv2.ellipse(im, (60 + 95 * i, 300), (20, (heights[i] if heights else 110) // 2), 0, 0, 360, col, -1)
    return im


def run(frames: list[np.ndarray], **settings):
    """(eye batch size per decision, indices of the covered people per decision, layer ids per decision, analyzer)."""
    an = stub_analyzer(Config.from_env(env={}, mode="offline", **settings), 1.0)
    sizes, covered, ids = [], [], []
    embed = an.models.eye.embed

    def counted(crops):
        sizes[-1] += len(crops)
        return embed(crops)

    an.models.eye.embed = counted
    for f, im in enumerate(frames):
        sizes.append(0)
        d = an(f * 3, im, cut=f == 0, now=f / 10)
        assert d.ok, an.stats["analyzer_errors"]
        drawn = {layer.tid for layer in d.layers}
        covered.append({round((box[0] + box[2] - 120) / 190) for tid, box in d.people if tid in drawn})
        ids.append({round((box[0] + box[2] - 120) / 190): tid for tid, box in d.people})
    return sizes, covered, ids, an


def test_budget_bounds_the_eye_and_nobody_is_shown_before_the_eye_saw_them():
    frames = [crowd(MIXED)] * 24
    risky = {i for i, c in enumerate(MIXED) if c == RED}
    free_sizes, free_cover, _, free = run(frames)
    assert max(free_sizes) == 9, "without a budget the whole crowd falls due in one decision"
    assert all(c == risky for c in free_cover)
    assert "eye_put_off" not in free.stats, "no budget: the scheduler stays out of it"

    sizes, covered, ids, an = run(frames, eye_budget=3)
    assert max(sizes) == 3, "never more crops than the budget"
    assert covered[0] == set(range(9)), "first decision: three seen, the six still waiting are covered"
    assert all(risky <= c for c in covered), "a risky person is covered in every decision, seen or not"
    assert all(c == risky for c in covered[6:]), "once everybody was seen, only the risky stay covered"
    assert sum(an.stats["eye_put_off"][8:]) == 0, "a crowd the budget can keep up with is not put off again"
    assert len(an.tracker.tracks) == 9
    waiting = [tid for tid in ids[0].values() if tid < 0]
    assert len(waiting) == 6 and len(set(waiting)) == 6, "people waiting for their first look have no track yet"
    still = [i for i, tid in ids[1].items() if tid < 0]
    assert still and all(ids[1][i] == ids[0][i] for i in still), "a waiting person keeps the same cover id"


def test_early_looks_spread_a_crowd_that_would_fall_due_together():
    frames = [crowd([GREEN] * 6)] * 20
    free_sizes, *_ = run(frames)
    sizes, covered, _, an = run(frames, eye_budget=6)  # room for everybody at once: nobody ever waits
    assert free_sizes[2:] == [0, 0, 6] * 6, "as before: nothing for two decisions, then everybody"
    assert max(sizes[2:]) == 2 and min(sizes[2:]) == 2, "with a budget: the crowd's average, every decision"
    assert sum(sizes) <= sum(free_sizes) + 6
    assert sum(an.stats["eye_put_off"]) == 0 and not any(covered)
    ages = [an.tracker.decision - t.eye_seen for t in an.tracker.tracks.values()]
    assert max(ages) <= an.cfg.eye.eye_every - 1, "no reading is older than EYE_EVERY allows"


def test_outfit_change_is_covered_as_fast_as_without_a_budget():
    """A known, uncovered person turns risky: seen (or covered while waiting) within EYE_EVERY - 1 decisions."""
    before, after = list(MIXED), list(MIXED)
    before[1], after[1] = BLUE, RED
    frames = [crowd(before)] * 14 + [crowd(after)] * 8
    for budget in (0, 3, 4):
        _, covered, *_ = run(frames, eye_budget=budget)
        assert 1 not in covered[13]
        first = next(f for f in range(14, 22) if 1 in covered[f])
        assert first <= 14 + 2, f"budget {budget}: covered at decision {first}"
        assert all(1 in c for c in covered[first:])


def test_a_cover_is_never_released_without_a_look():
    """One big covered person who turned modest, in a crowd that uses the whole budget: her look waits behind the
    others (she is covered anyway) for at most EYE_MAX_AGE decisions; decisions without a look never count toward
    her release. Without that limit she is never looked at again - and so never released."""
    heights = [170] + [110] * 6
    frames = [crowd([RED] + [GREEN] * 6, heights)] * 12 + [crowd([BLUE] + [GREEN] * 6, heights)] * 138
    _, free_cover, *_ = run(frames)
    free_release = next(f for f in range(12, 150) if 0 not in free_cover[f])
    assert free_release - 12 >= 10, "the release rule itself: 1 s of clear evidence"

    sizes, covered, *_ = run(frames, eye_budget=2)
    assert max(sizes) == 2
    release = next((f for f in range(12, 150) if 0 not in covered[f]), None)
    assert release is not None, "EYE_MAX_AGE gets her a look in the end"
    assert release > free_release, "later than with a look every time, never earlier"
    assert all(0 in c for c in covered[:release])

    _, covered, *_ = run(frames, eye_budget=2, eye_max_age=10**6)
    assert all(0 in c for c in covered), "no look, no release"


@pytest.mark.parametrize("mode", ["offline", "realtime"])
def test_budget_end_to_end_keeps_the_walker_covered(tmp_path, stub_run, mode):
    """Through the runner and the output gate, one crop per decision: the risky walker is covered in every frame;
    the safe person is covered only while waiting for the first look (start of the clip, after the cut)."""
    path = str(tmp_path / "walk.mp4")
    truth = make_video(path, fold_shade=0.9)  # whole blobs: two people of one size, the walker is found first
    report, sink = stub_run(path, mode, eye_budget=1)
    src = source_frames(path)
    s = report["summary"]
    assert s["analyzer_errors"] == 0 and s["failclosed_frames"] == 0
    assert report["embedded"]["max"] == 1
    red = [changed(o, f)[m].mean() for o, f, (m, _, _) in zip(sink.frames, src, truth, strict=True)]
    green = [changed(o, f)[g].mean() for o, f, (_, g, _) in zip(sink.frames, src, truth, strict=True)]
    assert min(red[3:]) >= 0.97, "the walking person must stay covered in every frame"
    waiting = [i for i, g in enumerate(green) if g > 0.05]
    assert waiting, "the safe person was covered until the eye had seen them"
    assert all(i < 12 or 57 <= i < 72 for i in waiting), f"...and only then: {waiting}"


# ------------------------------------------------------------------------------ risk policy rules
def _person(p: float, emb: np.ndarray, put_off: bool = False, waited: int = 0) -> Person:
    mask = np.zeros((540, 960), bool)
    mask[100:300, 100:160] = True
    return Person(box=(100, 100, 160, 300), mask=mask, emb=emb, p=p, put_off=put_off, waited=waited)


def _step(policy: RiskPolicy, tr: Tracker, person: Person, now: float) -> bool:
    if person.put_off:  # as the scheduler does: identity and last reading from the track
        tr.carry(person, 0)
    tr.assign([person], cut=False)
    policy.update([person], tr, THR, now=now)
    return person.risky


def test_put_off_decisions_do_not_count_toward_a_release():
    emb = np.zeros(8, np.float32)
    emb[0] = 1
    reading = {"p": 0.5}
    tr = Tracker()
    policy = RiskPolicy(RiskConfig(release_s=1.0), THR, lambda e: np.full(len(e), reading["p"]))
    t = 0.0
    for _ in range(5):
        assert _step(policy, tr, _person(0.5, emb), t)
        t += 0.1
    reading["p"] = 0.06
    seen = 0
    while True:  # one real (low) reading, then five decisions without a look, again and again
        risky = _step(policy, tr, _person(0.06, emb), t)
        seen += 1
        t += 0.1
        if not risky:
            break
        for _ in range(5):
            state = (tr.tracks[0].risk.evidence, tr.tracks[0].risk.safe_run, tr.tracks[0].risk.n_obs)
            assert _step(policy, tr, _person(0.0, emb, put_off=True), t), "covered while the look is put off"
            assert state == (tr.tracks[0].risk.evidence, tr.tracks[0].risk.safe_run, tr.tracks[0].risk.n_obs)
            t += 0.1

    tr2 = Tracker()  # the same readings with a look every time
    policy2 = RiskPolicy(RiskConfig(release_s=1.0), THR, lambda e: np.full(len(e), reading["p"]))
    reading["p"] = 0.5
    for i in range(5):
        _step(policy2, tr2, _person(0.5, emb), i / 10)
    reading["p"] = 0.06
    seen2 = next(n for n in range(1, 99) if not _step(policy2, tr2, _person(0.06, emb), (4 + n) / 10))
    assert seen >= seen2, "as many real low readings as without the waiting: the waiting itself released nothing"


def test_waiting_counts_toward_covering_only():
    """p 0.25: above the threshold, below the first-look bar of a big person - covered by evidence after 7 decisions.
    Someone who waited 6 decisions for the first look is covered at that look; a low reading counts once."""
    emb = np.zeros(8, np.float32)
    emb[0] = 1

    def decisions_until_covered(waited: int, p: float = 0.25) -> int | None:
        tr = Tracker()
        policy = RiskPolicy(RiskConfig(), THR, lambda e: np.full(len(e), p))
        return next((n for n in range(1, 12) if _step(policy, tr, _person(p, emb, waited=waited if n == 1 else 0), n / 10)), None)

    assert decisions_until_covered(0) == 7
    assert decisions_until_covered(6) == 1, "waited 6 decisions + this one = the 7 it takes"
    assert decisions_until_covered(3) == 4, "waited 3: covered 3 decisions sooner, at the same moment as without the wait"
    assert decisions_until_covered(10**6) == 1

    tr, tr_waited = Tracker(), Tracker()
    policy = RiskPolicy(RiskConfig(), THR, lambda e: np.full(len(e), 0.02))
    _step(policy, tr, _person(0.02, emb), 0.0)
    _step(policy, tr_waited, _person(0.02, emb, waited=9), 0.0)
    assert tr.tracks[0].risk.evidence == tr_waited.tracks[0].risk.evidence < 0, "safe evidence is never multiplied"
