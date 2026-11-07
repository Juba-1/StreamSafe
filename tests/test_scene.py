"""Scene layer: whole-screen cover for explicit scenes, the game-audio mute that goes with it, the OBS link."""

from __future__ import annotations

import numpy as np
import pytest

from streamsafe.analysis.scene import SceneLatch, hud_hit
from streamsafe.config import RenderConfig, SceneConfig
from streamsafe.io.obs_link import ObsStateWriter
from streamsafe.render.slate import SCENE_BACKGROUND
from streamsafe.runtime.gate import OutputGate
from streamsafe.types import Decision
from tests.stubs import make_video

W, H = 1920, 1080


def is_card(frame: np.ndarray) -> bool:
    return float(np.abs(frame.astype(int) - np.array(SCENE_BACKGROUND)).sum(2).mean()) < 30


def test_hud_keywords_survive_ocr_spacing():
    assert hud_hit("Press E to  End the Dance") == "endthedance"
    assert hud_hit("Follow her to the PRIVATE-ROOM.") == "privateroom"
    assert hud_hit("Press E to enter the vehicle") == ""


def test_latch_hysteresis_and_hold():
    latch = SceneLatch(SceneConfig(scene_enter=0.5, scene_exit=0.3, scene_hold_s=1.0, hud_hold_s=2.0))
    assert not latch.update(0.0, 0.4, None), "below the enter level"
    assert latch.update(0.5, 0.6, None)
    assert latch.update(1.0, 0.35, None), "above the exit level keeps it on"
    assert latch.update(1.5, None, None), "no reading: held"
    assert latch.update(2.0, 0.1, None), "low reading, but the last trigger (1.0) is within the hold"
    assert not latch.update(2.1, 0.1, None)
    assert not latch.update(2.2, 0.4, None), "off: 0.4 is below the enter level"
    assert latch.update(3.0, None, 3.0), "HUD hit"
    assert latch.update(4.9, 0.0, None) and not latch.update(5.1, 0.0, None), "HUD hold 2 s"


def test_gate_scene_card_covers_back_and_mutes_with_lead_and_tail():
    gate = OutputGate(RenderConfig(motion=False), (W, H), scene=SceneConfig(scene_back_s=0.1, mute_lead_s=0.1, mute_tail_s=0.1), fps=30)
    for k in range(0, 60, 3):
        gate.add_decision(Decision(frame_id=k, ok=True, scene=30 <= k < 45))
    frames = [np.full((H, W, 3), 100, np.uint8) for _ in range(60)]
    shown = [gate.render(v, frames[v]) for v in range(60)]
    gate.finish()
    card = [v for v, o in enumerate(shown) if o is gate.card]
    assert card[0] == 27, "3 frames (0.1 s) before the first scene decision are covered too"
    assert card[-1] == 44, "decision 42 is a scene decision: frames up to 44 use it"
    assert gate.stats["mute_intervals"] == [[24, 47]], "audio muted 3 frames before the card and 3 after"


@pytest.mark.parametrize("mode", ["offline", "realtime"])
def test_explicit_scene_gets_the_card_and_the_mute(tmp_path, stub_run, mode):
    path = str(tmp_path / "scene.mp4")
    make_video(path, n=150, cut_at=10**6, scene=(50, 80))
    report, sink = stub_run(path, mode, scene_model=True, scene_hold_s=0.5, scene_every_s=0.2)
    s = report["summary"]
    assert s["analyzer_errors"] == 0 and s["failclosed_frames"] == 0
    card = [v for v, o in enumerate(sink.frames) if is_card(o)]
    assert card, "the explicit scene is hidden"
    assert set(range(50, 80)) <= set(card), "every frame of the scene is covered"
    assert max(card) <= 80 + 15 + 9 + 6, "released after the hold (0.5 s) plus one scene period"
    assert min(card) >= 50 - 15 - 3, "not covered long before the scene"
    a, b = report["mute_intervals"][0]
    assert a <= min(card) and b >= max(card), "audio muted for the whole card"


def test_obs_state_file(tmp_path):
    path = tmp_path / "ss" / "state.txt"
    w = ObsStateWriter(str(path))
    w.update(5, muted=False, full=False)
    assert path.read_text() == "1 5 0 0\n"
    w.update(6, muted=False, full=False)
    assert path.read_text() == "1 5 0 0\n", "unchanged state is not rewritten every frame"
    w.update(7, muted=True, full=True)
    assert path.read_text() == "1 7 1 1\n", "a change is written at once"
    w.close()
    assert path.read_text().startswith("0 "), "a clean stop tells OBS to fail closed"


def test_late_hud_evidence_covers_back_to_its_own_frame():
    """A HUD read of frame 20 finishes at decision 45: frames 20.. are covered (they are still in the delay)."""
    gate = OutputGate(RenderConfig(motion=False), (W, H), scene=SceneConfig(scene_back_s=0.0, mute_lead_s=0.0, mute_tail_s=0.0), fps=30)
    for k in range(0, 90, 3):
        gate.add_decision(Decision(frame_id=k, ok=True, scene=45 <= k < 60, scene_from=20 if k == 45 else None))
    shown = [gate.render(v, np.full((H, W, 3), 100, np.uint8)) for v in range(90)]
    card = [v for v, o in enumerate(shown) if o is gate.card]
    assert card[0] == 20 and card[-1] == 59


def test_hud_thread_reads_in_background_and_counts_errors():
    import time

    from streamsafe.analysis.hud_thread import HudThread

    calls = []

    def reader(bgr):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("ocr broke")
        time.sleep(0.05)
        return "Press E to end the dance"

    hud = HudThread(reader)
    frame = np.zeros((10, 10, 3), np.uint8)
    assert hud.submit(1, 0.0, frame)
    assert not hud.submit(2, 0.1, frame), "busy: the frame is skipped, the caller never waits"
    t0 = time.monotonic()
    while not hud.idle() and time.monotonic() - t0 < 2:
        time.sleep(0.01)
    assert hud.results() == [(1, 0.0, "endthedance")]
    assert hud.submit(3, 0.2, frame)
    while not hud.idle() and time.monotonic() - t0 < 2:
        time.sleep(0.01)
    assert hud.results() == [(3, 0.2, "")] and hud.errors == 1


def test_body_close_up_gets_the_card(tmp_path, stub_run):
    """A risky person filling ~45% of the frame (frames 30-59) is hidden whole-screen, not with a giant garment."""
    import cv2

    from tests.stubs import RED

    path = str(tmp_path / "close.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    for f in range(120):
        im = np.full((1080, 1920, 3), (90, 120, 140), np.uint8)
        axes = (700, 560) if 30 <= f < 60 else (70, 230)
        cv2.ellipse(im, (960, 540), axes, 0, 0, 360, RED, -1)
        writer.write(im)
    writer.release()
    report, sink = stub_run(path, "offline", close_hold_s=0.3)
    card = [v for v, o in enumerate(sink.frames) if is_card(o)]
    assert set(range(30, 60)) <= set(card), "the close-up is hidden"
    assert min(card) >= 30 - 15, "starts at most scene_back_s (0.5 s) before the first close decision"
    assert max(card) <= 60 + 9 + 3, "released after the close hold (0.3 s); the small person gets a garment only"
