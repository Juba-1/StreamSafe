"""Stutter evidence and guards: encoder speed, output queue, preview off the stream thread."""

import time

import numpy as np

from streamsafe import session_report
from streamsafe.io import sinks
from streamsafe.runtime.runner import Snapper


def test_encoder_speed_counts_only_once_connected():
    s = sinks.FFmpegSink("rtmp://x/app/k", (64, 36), 30.0, spawn=lambda cmd, **kw: None)
    s._speed("0.5x")  # still connecting: not counted
    s.connected = True
    for text in ("1.0x", "0.90x", "N/A", "1.01x"):
        s._speed(text)
    assert (s.speed_reports, s.slow_reports) == (3, 1)


def test_live_presets_are_lighter_than_recording():
    assert sinks.encoder_flags("h264_nvenc", True)[1] == "p4" and sinks.encoder_flags("h264_nvenc", False)[1] == "p5"
    assert "balanced" in sinks.encoder_flags("h264_amf", True) and "faster" in sinks.encoder_flags("h264_qsv", True)


class Slow:
    def write(self, frame):
        time.sleep(0.02)

    def close(self):
        pass


def test_threaded_sink_measures_waiting():
    t = sinks.ThreadedSink(Slow(), max_queue=2)
    for _ in range(8):
        t.write(np.zeros((2, 2, 3), np.uint8))
    t.close()
    assert t.queue_max == 2 and t.blocked_s > 0


def test_report_names_a_slow_encoder():
    r = session_report.build({"summary": {"frames_out": 10, "encoder_slow_share": 0.2}}, 30)
    assert "encoder_slow" in r["problems"] and session_report.AR_PROBLEMS["encoder_slow"]


def test_preview_never_blocks(tmp_path):
    path = str(tmp_path / "p.png")
    snap = Snapper(path)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    t = time.perf_counter()
    for _ in range(20):
        snap(frame)
    assert time.perf_counter() - t < 0.05
    for _ in range(100):
        if (tmp_path / "p.png").exists():
            break
        time.sleep(0.02)
    assert (tmp_path / "p.png").exists()
