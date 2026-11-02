"""Failure injection: the viewer must never see an unprotected frame."""

from __future__ import annotations

from streamsafe.config import Config
from streamsafe.io.sources import VideoFileSource
from streamsafe.runtime.runner import run
from tests.conftest import MemorySink
from tests.crashing import CRASH_AT, process_dies
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
