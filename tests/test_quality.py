"""The protection must not make the picture worse than what came in (Ali, 17 Nov 2026): the size and frame rate
of the source stay, the bitrate fits the size and the platform, the encoders run at their quality settings."""

import json
import subprocess
from pathlib import Path

import pytest

from streamsafe import cli
from streamsafe.app import engine, ingest, plan
from streamsafe.app import settings as cfg
from streamsafe.config import Config
from streamsafe.io import sinks
from streamsafe.io.sources import sender_format
from streamsafe.runtime.cover_all import cover_all_analyzer


def test_rules_in_frames_keep_their_time_at_60_fps():
    c = Config().at_fps(60)
    assert (c.runtime.fps, c.runtime.step_every, c.render.max_span, c.render.max_gap, c.render.hold, c.render.lookahead_frames) == (
        60.0, 6, 18, 36, 4, 120,
    )  # fmt: skip
    c = Config().at_fps(30)
    assert (c.runtime.step_every, c.render.max_span) == (3, 9)  # unchanged at the rate they were tuned at
    assert Config().at_fps(240).runtime.fps == 60.0  # at most 60


def test_bitrates_by_size_and_platform():
    assert sinks.auto_kbps((1920, 1080), 30, network=True) == 6000
    assert sinks.auto_kbps((1920, 1080), 60, network=True) == 9000
    assert sinks.auto_kbps((1920, 1080), 60, network=False) == 24883  # a recording: ~0.2 bit per pixel and frame
    assert sinks.auto_kbps((3840, 2160), 60, network=False) == 99533
    assert sinks.auto_kbps((1280, 720), 30, network=False) == 20000  # never below 20 Mbit/s
    assert ingest.bitrate_for("kick", 1080, 60) == 8000 and ingest.bitrate_for("twitch", 1080, 60) == 6000
    assert ingest.bitrate_for("youtube", 2160, 60) == 51000 and ingest.bitrate_for("youtube", 1440, 30) == 24000


def _cmd(url, size=(1920, 1080), fps=30.0, **kw):
    return sinks.FFmpegSink(url, size, fps, spawn=lambda *a, **k: None, **kw)._cmd


def test_stream_and_recording_settings(monkeypatch):
    monkeypatch.delenv("VIDEO_KBPS", raising=False)
    monkeypatch.delenv("ENCODER_PLAIN", raising=False)
    live = _cmd("rtmp://h/app/k", encoder="h264_nvenc")
    assert live[live.index("-b:v") + 1] == "6000k" and live[live.index("-maxrate") + 1] == "6000k"
    assert live[live.index("-rc") + 1] == "cbr" and live[live.index("-preset") + 1] == "p5" and "-tune" in live
    monkeypatch.setenv("VIDEO_KBPS", "8000")  # what the app sets for the platform
    assert _cmd("rtmp://h/app/k")[_cmd("rtmp://h/app/k").index("-b:v") + 1] == "8000k"
    monkeypatch.delenv("VIDEO_KBPS")
    f = _cmd("out.mkv", (3840, 2160), 60.0, encoder="h264_nvenc", audio_bitrate="320k")
    assert f[f.index("-rc") + 1] == "vbr" and f[f.index("-b:v") + 1] == "99533k" and f[f.index("-b:a") + 1] == "320k"
    assert f[f.index("-g") + 1] == "120"  # a keyframe every 2 s at 60 fps
    monkeypatch.setenv("ENCODER_PLAIN", "1")  # the encoder refused the quality settings on this PC
    assert "-preset" not in _cmd("rtmp://h/app/k", encoder="h264_nvenc")


def test_what_obs_sends_is_read():
    line = (
        "  Stream #0:0[0x100]: Video: h264 (High), yuv420p(tv, bt709, progressive), 2560x1440 [SAR 1:1 DAR 16:9], "
        "30000 kb/s, 60 fps, 60 tbr, 90k tbn"
    )
    assert sender_format(line) == {"width": 2560, "height": 1440, "fps": 60.0, "kbps": 30000}
    assert sender_format("  Stream #0:1[0x101]: Audio: aac (LC), 48000 Hz, stereo, fltp, 160 kb/s") is None


def test_run_args_carry_quality_and_delay(tmp_path):
    s = cfg.Settings(input=cfg.INPUT_SRT, output=cfg.OUTPUT_RTMP, platform="kick", resolution="1440p", fps=60)
    a = engine.run_args(s, tmp_path, fast=True)
    get = lambda name: a[a.index(name) + 1]  # noqa: E731
    assert get("--size") == "1920x1080"  # Kick takes at most 1080p: 1440p would only be scaled down by Kick
    assert (get("--fps"), get("--bitrate"), get("--delay")) == ("60", "8000", "2")
    a = engine.run_args(
        cfg.Settings(input=cfg.INPUT_SRT, output=cfg.OUTPUT_RTMP, platform="youtube", resolution="1440p", fps=60), tmp_path, fast=False
    )
    assert a[a.index("--size") + 1] == "2560x1440" and a[a.index("--delay") + 1] == "3"  # slower PC: one more second
    a = engine.run_args(cfg.Settings(input=cfg.INPUT_FILE, input_file="v.mp4", output=cfg.OUTPUT_RECORD, delay=4.0), tmp_path, "o.mp4")
    assert "--size" not in a and a[a.index("--fps") + 1] == "auto" and a[a.index("--delay") + 1] == "4"


def test_plan_gates_the_custom_logo(tmp_path):
    env = {"STREAMSAFE_DATA_DIR": str(tmp_path)}
    assert plan.current(env) == plan.BASIC and not plan.allows("custom_brand", env)
    (tmp_path / "license.json").write_text(json.dumps({"plan": "pro"}))  # a hand-written file is not a licence
    assert not plan.allows("custom_brand", env)
    assert plan.allows("custom_brand", {**env, "STREAMSAFE_PLAN": "pro"})
    assert plan.current({**env, "STREAMSAFE_PLAN": "basic"}) == plan.BASIC


def _clip(path: Path, size: str, rate: int, seconds: float = 1.0) -> str:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", str(path)],
        check=True, timeout=60,
    )  # fmt: skip
    return str(path)


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], capture_output=True, text=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


@pytest.mark.parametrize("size,rate", [("1920x1080", 60), ("2560x1440", 30)])
def test_a_recorded_video_keeps_its_size_and_frame_rate(tmp_path, size, rate):
    clip = _clip(tmp_path / "in.mp4", size, rate)
    out = tmp_path / "out.mp4"
    code = cli.main(
        ["--input", f"file:{clip}", "--output", f"record:{out}", "--mode", "offline", "--fps", "auto", "--report-dir", str(tmp_path)],
        factory=cover_all_analyzer,
    )
    assert code == 0
    v = _probe(out)["video"]
    assert f"{v['width']}x{v['height']}" == size
    num, den = (int(x) for x in v["avg_frame_rate"].split("/"))
    assert abs(num / den - rate) < 0.5
    assert cli.input_fps(f"file:{clip}", "auto") == rate and cli.input_fps("srt://x:1", "auto") == 0.0
