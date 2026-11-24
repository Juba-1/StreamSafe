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


def test_vertical_cut_follows_the_live_choice():
    from streamsafe.runtime.control import Live

    class Ctl:
        now = Live()

        def get(self):
            return self.now

    got = []

    class Out:
        def write(self, frame):
            got.append(frame.shape)

        def close(self):
            pass

    size = sinks.vertical_size((1920, 1080))
    assert size == (608, 1080)
    ctl = Ctl()
    v = sinks.VerticalSink(Out(), size, ctl)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert v.box(1920, 1080) == (656, 0, 608, 1080)  # the middle, full height
    ctl.now = Live(vcrop=(0.7, 0.5, 0.9, 1.0))  # lower right, half height: scaled to the fixed size
    x, y, cw, ch = v.box(1920, 1080)
    assert (y, ch, cw) == (540, 540, 304) and x + cw <= 1920
    v.write(frame)
    assert got == [(1080, 608, 3)]


def test_always_covered_parts():
    from streamsafe.runtime.control import parse
    from streamsafe.runtime.runner import COVER_BGR, cover_parts

    live = parse({"covers": [[0, 0, 0.5, 0.5], [2, 2, 3, 3]], "vcrop": [0.3, 0, 0.6, 1]})
    assert live.covers == ((0, 0, 0.5, 0.5),) and live.vcrop == (0.3, 0, 0.6, 1)
    f = np.full((100, 200, 3), 255, np.uint8)
    cover_parts(f, live.covers)
    assert tuple(f[10, 10]) == COVER_BGR and tuple(f[90, 190]) == (255, 255, 255)
