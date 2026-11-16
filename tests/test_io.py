"""Sources and sinks through FFmpeg, including an SRT round trip like OBS -> StreamSafe."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from streamsafe.config import Config
from streamsafe.io.sinks import FFmpegSink, ThreadedSink
from streamsafe.io.sources import FFmpegSource, VideoFileSource, open_source
from streamsafe.runtime.runner import run
from tests.stubs import make_video, stub_analyzer

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _has_srt() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-protocols"], capture_output=True, text=True).stdout
    return " srt\n" in out or "\n  srt\n" in out


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_open_source_dispatch(tmp_path):
    path = str(tmp_path / "v.mp4")
    make_video(path, n=5)
    src = open_source(path)
    assert isinstance(src, VideoFileSource) and src.size == (1920, 1080) and not src.live
    assert len(list(src.frames())) == 5


@needs_ffmpeg
def test_ffmpeg_sink_then_source_round_trip(tmp_path):
    out = str(tmp_path / "out.mp4")
    sink = ThreadedSink(FFmpegSink(out, (640, 360), 30.0))
    for i in range(30):
        sink.write(np.full((360, 640, 3), (i * 8) % 255, np.uint8))
    sink.close()
    frames = list(FFmpegSource(out, (640, 360), 30.0).frames())
    assert len(frames) == 30
    assert frames[0].shape == (360, 640, 3)


@needs_ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None or not _has_srt(), reason="ffmpeg without SRT")
def test_srt_input_realtime_pipeline(tmp_path):
    """OBS-like sender -> SRT -> StreamSafe (realtime) -> H.264 file."""
    video = str(tmp_path / "walk.mp4")
    make_video(video, n=90, cut_at=10**6)
    port = _free_port()
    source = FFmpegSource(f"srt://127.0.0.1:{port}?mode=listener", (1920, 1080), 30.0, relisten=False)  # ends with the sender
    time.sleep(1.0)
    sender = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-i", video, "-c:v", "libx264", "-preset", "ultrafast",
         "-tune", "zerolatency", "-f", "mpegts", f"srt://127.0.0.1:{port}?mode=caller"],
    )  # fmt: skip
    out = str(tmp_path / "protected.ts")
    sink = ThreadedSink(FFmpegSink(out, source.size, 30.0))
    try:
        report = run(Config.from_env(env={}, mode="realtime", delay=1.0), source, sink, factory=stub_analyzer)
    finally:
        sender.wait(timeout=60)
        source.close()
        sink.close()
    s = report["summary"]
    assert s["frames_in"] >= (30 if os.environ.get("CI") else 80)  # shared CI runners are slow; locally the full rate
    assert s["frames_out"] + s["dropped_frames"] == s["frames_in"]
    assert s["frames_out"] >= 60  # a 2-core CI box may drop a few late frames (never shows them late)
    assert s["analyzer_errors"] == 0
    assert s["decisions"] > 3  # 2-core CI box shared with two ffmpeg encoders
    # picture only: with sound, the reader starts the picture at the sound's first instant (one repeated frame)
    n_written = sum(1 for _ in FFmpegSource(out, (1920, 1080), 30.0, audio=False).frames())
    assert n_written == s["frames_out"]


def test_ffmpeg_input_args_and_60fps_input_becomes_30fps():
    """A 60 fps feed (OBS virtual camera at 60) is resampled to the engine's 30 fps, so frame counts and the
    clock agree (the delay buffer is sized in frames)."""
    src = FFmpegSource("testsrc=size=1280x720:rate=60:duration=2", (640, 360), 30.0, input_args=("-f", "lavfi"), live=False)
    n = sum(1 for _ in src.frames())
    src.close()
    assert 58 <= n <= 61


def test_device_by_name_needs_windows():
    if sys.platform == "win32":
        pytest.skip("Windows opens devices by name through FFmpeg dshow")
    with pytest.raises(ValueError, match="Windows"):
        open_source("device:OBS Virtual Camera")


def test_window_is_black_at_once_stays_responsive_and_goes_black_when_frames_stop(monkeypatch):
    """The window lives on its own thread: black from the first moment (models still loading), keeps drawing while
    the engine is busy (Windows never marks it "Not Responding"), and goes black when frames stop (fail closed)."""
    import time as _t

    from streamsafe.io import sinks

    shown, waits = [], []
    monkeypatch.setattr(sinks.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(sinks.cv2, "imshow", lambda title, f: shown.append(f.copy()))
    monkeypatch.setattr(sinks.cv2, "waitKey", lambda ms: waits.append(ms) or _t.sleep(ms / 1000) or -1)
    monkeypatch.setattr(sinks.cv2, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(sinks.WindowSink, "STALE_S", 0.2)
    w = sinks.WindowSink("t")
    assert shown and not shown[0].any()  # black before any frame
    _t.sleep(0.3)  # the engine is busy: the window keeps pumping events anyway
    assert len(waits) >= 10
    frame = np.full((72, 128, 3), 200, np.uint8)
    w.write(frame)
    _t.sleep(0.1)
    assert any(f.shape == frame.shape and f.mean() == 200 for f in shown)
    _t.sleep(0.4)  # no new frames: black at the same size
    assert shown[-1].shape == frame.shape and not shown[-1].any()
    w.close()
    assert not w._thread.is_alive()
