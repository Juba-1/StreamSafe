"""Sources and sinks through FFmpeg, including an SRT round trip like OBS -> StreamSafe."""

from __future__ import annotations

import shutil
import socket
import subprocess
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
    source = FFmpegSource(f"srt://127.0.0.1:{port}?mode=listener", (1920, 1080), 30.0)
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
    assert s["frames_in"] >= 80
    assert s["frames_out"] + s["dropped_frames"] == s["frames_in"]
    assert s["frames_out"] >= 60  # a 2-core CI box may drop a few late frames (never shows them late)
    assert s["analyzer_errors"] == 0
    assert s["decisions"] > 3  # 2-core CI box shared with two ffmpeg encoders
    n_written = sum(1 for _ in FFmpegSource(out, (1920, 1080), 30.0).frames())
    assert n_written == s["frames_out"]
