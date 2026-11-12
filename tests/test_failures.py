"""Failure injection: the viewer must never see an unprotected frame."""

from __future__ import annotations

from streamsafe.config import Config
from streamsafe.io.sources import VideoFileSource
from streamsafe.runtime.runner import run
from tests.conftest import MemorySink
from tests.crashing import CRASH_AT, process_dies, process_dies_once
from tests.stubs import make_video
from tests.test_engine import is_slate


def test_analyzer_process_death_fails_closed(tmp_path):
    path = str(tmp_path / "walk.mp4")
    make_video(path, n=150, cut_at=10**6)
    cfg = Config.from_env(env={}, mode="realtime", delay=1.0)
    source, sink = VideoFileSource(path, 30.0), MemorySink()
    report = run(cfg, source, sink, factory=process_dies)
    assert report["summary"]["frames_out"] == 150, "the stream keeps running"
    tail = sink.frames[CRASH_AT + 25 :]  # decisions stop arriving: max_gap + hold later every frame is the slate
    assert tail and all(is_slate(f) for f in tail)


def test_analyzer_is_replaced_after_a_crash(tmp_path, monkeypatch):
    """A crashed analyzer (GPU driver reset) is started again: slate while it loads, protection after."""
    monkeypatch.setenv("STREAMSAFE_TEST_FLAG", str(tmp_path / "crashed"))
    path = str(tmp_path / "walk.mp4")
    make_video(path, n=450, cut_at=10**6)
    cfg = Config.from_env(env={}, mode="realtime", delay=1.0)
    source, sink = VideoFileSource(path, 30.0), MemorySink()
    report = run(cfg, source, sink, factory=process_dies_once)
    assert report["summary"]["worker_restarts"] == 1
    slate = [is_slate(f) for f in sink.frames]
    first_slate = slate.index(True)
    assert first_slate >= CRASH_AT, "protected normally before the crash"
    back = [i for i in range(first_slate, len(slate)) if not slate[i]]
    assert back, "protection resumed after the restart"
    assert all(not s for s in slate[back[0] + 30 :]), "and stayed on"


class SlowSink(MemorySink):
    """An output slower than the stream (an encoder that cannot keep up)."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds

    def write(self, frame) -> None:
        import time

        time.sleep(self.seconds)
        super().write(frame)


def test_slow_output_drops_frames_instead_of_drifting(tmp_path):
    """Run 44 soak: an output slower than 30 fps made frames pile up (40 GB, minutes of extra delay).
    Now late frames are dropped: the delay stays bounded and every frame shown is still protected."""
    path = str(tmp_path / "walk.mp4")
    make_video(path, n=180, cut_at=10**6)
    cfg = Config.from_env(env={}, mode="realtime", delay=0.5, max_late_s=0.2)
    source, sink = VideoFileSource(path, 30.0), SlowSink(0.05)  # 20 fps out for a 30 fps stream
    from tests.stubs import stub_analyzer

    report = run(cfg, source, sink, factory=stub_analyzer)
    s = report["summary"]
    assert s["frames_in"] == 180
    assert s["dropped_frames"] > 30, "a third of the frames cannot be shown in time"
    assert s["frames_out"] == len(sink.frames) == 180 - s["dropped_frames"]
    assert report["emit_lateness_ms"]["max"] < 200 + 50 + 100, "max_late_s + one slow write + scheduling; never grows"


class InterruptingSink(MemorySink):
    """The streamer presses Ctrl+C after some frames."""

    def write(self, frame) -> None:
        super().write(frame)
        if len(self.frames) == 40:
            raise KeyboardInterrupt


def test_ctrl_c_still_closes_cleanly_and_reports(tmp_path):
    path = str(tmp_path / "walk.mp4")
    make_video(path, n=150, cut_at=10**6)
    cfg = Config.from_env(env={}, mode="realtime", delay=0.5)
    from tests.stubs import stub_analyzer

    report = run(cfg, VideoFileSource(path, 30.0), InterruptingSink(), factory=stub_analyzer)
    assert report["summary"]["stopped_by_user"] is True
    assert report["summary"]["frames_out"] == 39, "the frame being written when Ctrl+C came is not counted as shown"
