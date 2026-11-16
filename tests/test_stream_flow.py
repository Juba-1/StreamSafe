"""The live path end to end without models: the window's controller around a failing / working engine, the
engine's command line with the whole-screen stand-in analyzer, recordings, and failed-start reports."""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from streamsafe import cli, session_report
from streamsafe.app import controller, engine, heads, modelpack, updates
from streamsafe.app import settings as cfg
from streamsafe.io import link
from streamsafe.runtime.cover_all import cover_all_analyzer
from tests.appfix import FakeProtector

FAKE = str(Path(__file__).with_name("fake_engine.py"))
KEY = "sk_us-west-2_TESTkey_1234567890"


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    env = {"STREAMSAFE_DATA_DIR": str(tmp_path / "data")}
    monkeypatch.setattr(engine, "exe_command", lambda args: [sys.executable, FAKE, *args])
    monkeypatch.setattr(modelpack, "problem", lambda models, quick=True: "")
    monkeypatch.setattr(heads, "why_not", lambda env=None, app_folder=None: "")
    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: False)
    monkeypatch.delenv("OBS_STATE", raising=False)
    c = controller.Controller(env, FakeProtector(), log=lambda line: None)
    yield c
    c.engine.stop(2)


def _until_ended(c, seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end and (c.engine.running or c.engine.exit_code is None):
        c.status()
        time.sleep(0.05)
    c.status()


def test_busy_port_moves_and_obs_gets_the_new_address(ctrl):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        busy = s.getsockname()[1]
        ctrl.settings.input, ctrl.settings.srt_port = cfg.INPUT_SRT, busy
        assert ctrl.free_port() is True
    assert ctrl.port_moved_from == busy and ctrl.settings.srt_port != busy
    assert ctrl.obs_server() == f"srt://127.0.0.1:{ctrl.settings.srt_port}"
    assert ctrl.free_port() is False  # free now: stays


def test_kick_server_without_app_streams_to_the_right_address(ctrl, monkeypatch):
    seen = {}
    real = engine.EngineProcess.start

    def spy(self, cmd, env, log_path=None):
        seen["env"] = dict(env)
        return real(self, cmd, env, log_path)

    monkeypatch.setattr(engine.EngineProcess, "start", spy)
    ctrl.settings.output, ctrl.settings.platform = cfg.OUTPUT_RTMP, "kick"
    ctrl.settings.rtmp_url = "rtmps://fa723fc1b171.global-contribute.live-video.net/"
    ctrl.start(" " + KEY)
    assert seen["env"][engine.OUTPUT_ENV] == f"rtmps://fa723fc1b171.global-contribute.live-video.net:443/app/{KEY}"
    ctrl.stop()


def test_platform_refusal_is_explained_and_reported_with_the_log(ctrl, monkeypatch):
    monkeypatch.setenv(
        "FAKE_STATUS",
        '{"event": "waiting_obs", "port": 9000}|{"event": "obs_connected"}|{"event": "platform_connecting"}|'
        '{"event": "platform_failed", "reason": "refused_key", "detail": "NetStream.Publish.BadName"}|'
        '{"event": "stopped", "reason": "platform_failed", "detail": "refused_key"}',
    )
    monkeypatch.setenv("FAKE_EXIT", "8")
    ctrl.settings.output, ctrl.settings.platform = cfg.OUTPUT_RTMP, "twitch"
    ctrl.start(KEY)
    _until_ended(ctrl)
    assert ctrl.link.platform == "failed" and ctrl.link.reason == "refused_key"
    assert ctrl.exit_reason() == "engine_platform_failed"
    report = ctrl.last_report()
    assert report is not None
    text = report.read_text(encoding="utf-8")
    assert "platform_failed" in text and session_report.LOG_HEAD in text and KEY not in text


def test_engine_that_dies_at_once_still_leaves_a_report(ctrl, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "6")
    monkeypatch.setenv("FAKE_STATUS", "")
    ctrl.start("")
    _until_ended(ctrl)
    assert ctrl.exit_reason() == "engine_weak_gpu"
    report = ctrl.last_report()
    assert report is not None and "start_failed" in report.read_text(encoding="utf-8")


def test_run_args_for_every_input_and_output(tmp_path):
    s = cfg.Settings(input=cfg.INPUT_SRT, srt_port=9123, output=cfg.OUTPUT_RTMP)
    args = engine.run_args(s, tmp_path)
    assert args[args.index("--input") + 1] == "srt://127.0.0.1:9123?mode=listener"
    assert args[args.index("--output") + 1] == f"env:{engine.OUTPUT_ENV}"
    s = cfg.Settings(input=cfg.INPUT_CAMERA, camera="Elgato HD60 X", camera_sound="HD60 X Audio", output=cfg.OUTPUT_RECORD)
    args = engine.run_args(s, tmp_path, "C:/v/a.mp4")
    assert (
        args[args.index("--input") + 1] == "device:Elgato HD60 X|HD60 X Audio" and args[args.index("--output") + 1] == "record:C:/v/a.mp4"
    )
    s = cfg.Settings(input=cfg.INPUT_FILE, input_file="C:/v/game.mp4", output=cfg.OUTPUT_RECORD)
    assert engine.run_args(s, tmp_path, "C:/v/b.mp4")[2] == "file:C:/v/game.mp4"


def test_settings_problems_for_files(tmp_path):
    s = cfg.Settings(input=cfg.INPUT_FILE, input_file=str(tmp_path / "none.mp4"), output=cfg.OUTPUT_WINDOW)
    assert "bad_input_file" in s.problems() and "bad_file_output" in s.problems()


def _clip(path: Path, seconds: float = 1.5) -> str:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", str(path)],
        check=True, timeout=60,
    )  # fmt: skip
    return str(path)


def test_record_only_of_a_finished_video_is_fully_protected(tmp_path, capsys):
    """A creator's recorded video -> protected MP4 (here with the whole-screen stand-in: every frame covered)."""
    clip = _clip(tmp_path / "game.mp4")
    out = tmp_path / "rec" / "protected.mp4"
    code = cli.main(
        ["--input", f"file:{clip}", "--output", f"record:{out}", "--mode", "offline", "--report-dir", str(tmp_path)],
        factory=cover_all_analyzer,
    )
    assert code == 0 and out.is_file() and not (tmp_path / "rec" / "protected.recording.mkv").exists()
    events = [link.parse(x) for x in capsys.readouterr().out.splitlines() if x.startswith("STATUS ")]
    names = [e["event"] for e in events]
    assert "recording" in names and "recorded" in names and names[-1] == "stopped"
    cap = cv2.VideoCapture(str(out))
    n, bright = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        bright = max(bright, float(np.mean(frame)))
    cap.release()
    assert n >= 40
    src = cv2.VideoCapture(clip)
    _, first = src.read()
    src.release()
    assert bright < float(np.mean(first))  # nothing of the colourful source got through: the cover only
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)], capture_output=True, text=True)
    kinds = {s["codec_type"] for s in json.loads(probe.stdout)["streams"]}
    assert kinds == {"video", "audio"}


def test_srt_in_platform_out_with_obs_restart(tmp_path, capsys):
    """OBS-like sender over SRT (stops and starts again mid-run) -> engine -> a local SRT 'platform'."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port_in = s.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port_out = s.getsockname()[1]
    got = tmp_path / "platform.ts"
    platform = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", f"srt://127.0.0.1:{port_out}?mode=listener", "-c", "copy", str(got)]
    )
    clip = _clip(tmp_path / "game.mp4", 3)

    def send():
        return subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-i", clip, "-c:v", "libx264", "-preset", "ultrafast",
             "-c:a", "aac", "-f", "mpegts", f"srt://127.0.0.1:{port_in}?mode=caller"],
        )  # fmt: skip

    import threading

    def obs():
        time.sleep(1.0)
        send().wait(timeout=30)
        time.sleep(1.0)
        send().wait(timeout=30)
        time.sleep(4.0)
        import _thread

        _thread.interrupt_main()  # the streamer presses Stop

    threading.Thread(target=obs, daemon=True).start()
    code = cli.main(
        ["--input", f"srt://127.0.0.1:{port_in}?mode=listener", "--output", f"srt://127.0.0.1:{port_out}?mode=caller",
         "--delay", "1", "--size", "320x180", "--report-dir", str(tmp_path)],
        factory=cover_all_analyzer,
    )  # fmt: skip
    platform.wait(timeout=30)
    assert code in (0, 130)
    events = [link.parse(x) for x in capsys.readouterr().out.splitlines() if x.startswith("STATUS ")]
    names = [e["event"] for e in events]
    assert names.count("obs_connected") == 2 and "platform_connected" in names
    assert events[-1] == {"event": "stopped", "reason": "user", "detail": ""}
    assert got.is_file() and got.stat().st_size > 10_000
