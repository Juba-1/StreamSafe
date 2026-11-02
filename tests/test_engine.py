"""End-to-end behaviour with stand-in models, in both runtime modes."""

from __future__ import annotations

import numpy as np
import pytest

from tests.stubs import BLUE, GREEN, make_video

SLATE_BGR = np.array([40, 28, 22])


def changed(out: np.ndarray, src: np.ndarray) -> np.ndarray:
    return np.abs(out.astype(int) - src.astype(int)).sum(2) > 60


def source_frames(path: str) -> list[np.ndarray]:
    import cv2

    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA))


def is_slate(frame: np.ndarray) -> bool:
    return float(np.abs(frame.astype(int) - SLATE_BGR).sum(2).mean()) < 30


@pytest.mark.parametrize("mode", ["offline", "realtime"])
def test_walking_person_is_covered_and_safe_person_is_not(tmp_path, stub_run, mode):
    path = str(tmp_path / "walk.mp4")
    truth = make_video(path)
    report, sink = stub_run(path, mode)
    src = source_frames(path)
    s = report["summary"]
    assert s["analyzer_errors"] == 0
    assert s["failclosed_frames"] == 0
    assert s["frames_out"] == len(truth) == len(sink.frames)
    red = [changed(o, f)[m].mean() for o, f, (m, _, risky) in zip(sink.frames, src, truth, strict=True) if risky]
    green = [changed(o, f)[g].mean() for o, f, (_, g, _) in zip(sink.frames, src, truth, strict=True)]
    assert min(red[3:]) >= 0.97, "the walking person must stay covered in every frame"
    assert max(green) < 0.05, "a clearly safe person is never covered"
    assert not any(is_slate(o) for o in sink.frames)


@pytest.mark.parametrize(("colour", "lo", "hi"), [(GREEN, 0, 15), (BLUE, 30, 75)])
def test_release_after_turning_safe(tmp_path, stub_run, colour, lo, hi):
    """Very sure safe readings release at once; borderline-safe ones only after ~release_s."""
    path = str(tmp_path / "turn.mp4")
    truth = make_video(path, n=150, cut_at=10**6, turn_safe_at=60, turn_col=colour)
    _, sink = stub_run(path, "offline")
    src = source_frames(path)
    cover = np.array([changed(o, f)[m].mean() for o, f, (m, _, _) in zip(sink.frames, src, truth, strict=True)])
    assert cover[3:60].min() >= 0.97
    released = next((i for i in range(60, 150) if cover[i] < 0.05), None)
    assert released is not None
    assert lo <= released - 60 <= hi


def test_flat_colour_person_stays_covered(tmp_path, stub_run):
    """No texture: motion compensation must refuse to guess instead of throwing the garment away."""
    path = str(tmp_path / "flat.mp4")
    truth = make_video(path, texture=False)
    _, sink = stub_run(path, "offline")
    src = source_frames(path)
    red = [changed(o, f)[m].mean() for o, f, (m, _, risky) in zip(sink.frames, src, truth, strict=True) if risky]
    assert min(red[3:]) >= 0.97


@pytest.mark.parametrize("mode", ["offline", "realtime"])
def test_one_frame_flash_never_shows_the_slate(tmp_path, stub_run, mode):
    path = str(tmp_path / "flash.mp4")
    make_video(path, flash_at=40)
    report, sink = stub_run(path, mode)
    assert report["summary"]["failclosed_frames"] == 0
    assert not any(is_slate(o) for o in sink.frames)
    if mode == "offline":
        assert report["summary"]["held_frames"] >= 1


def test_non_1080p_input_keeps_garments_on_the_person(tmp_path, stub_run):
    """720p input: layers are scaled to the output size (output scale 1.33)."""
    path = str(tmp_path / "walk720.mp4")
    truth = make_video(path, size=(1280, 720), fold_shade=0.9)
    report, sink = stub_run(path, "offline")
    src = source_frames(path)
    red = [changed(o, f)[m].mean() for o, f, (m, _, risky) in zip(sink.frames, src, truth, strict=True) if risky]
    assert report["summary"]["failclosed_frames"] == 0
    assert min(red[3:]) >= 0.97
