"""The window's logic without a window: blockers, start / watch / stop of the engine process, key handling."""

import sys
import time
from pathlib import Path

import pytest

from streamsafe.app import controller, engine, heads, modelpack, paths, updates
from streamsafe.app import settings as cfg
from tests.appfix import FakeProtector

FAKE = str(Path(__file__).with_name("fake_engine.py"))
KEY = "live_987654321_SECRETKEYVALUE"


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    env = {"STREAMSAFE_DATA_DIR": str(tmp_path / "data")}
    monkeypatch.setattr(engine, "exe_command", lambda args: [sys.executable, FAKE, *args])
    monkeypatch.setattr(modelpack, "problem", lambda models, quick=True: "")
    monkeypatch.setattr(heads, "why_not", lambda env=None, app_folder=None: "")
    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: False)
    monkeypatch.delenv("OBS_STATE", raising=False)
    c = controller.Controller(env, FakeProtector(), log=lambda line: lines.append(line))
    c.settings.input, c.settings.output = "camera", "window"  # these tests drive the engine without OBS or a platform
    lines.clear()
    yield c
    c.engine.stop(2)


lines: list[str] = []


def wait_for(c, want, seconds=8.0):
    end = time.monotonic() + seconds
    got = ""
    while time.monotonic() < end:
        got = c.status()
        if got in want:
            return got
        time.sleep(0.05)
    raise AssertionError(f"status stayed {got!r}, wanted {want}")


def rtmp(c):
    c.settings.output = cfg.OUTPUT_RTMP
    c.settings.rtmp_url = "rtmps://live.example/app"


def test_start_streams_with_the_key_only_in_the_environment(ctrl, monkeypatch):
    seen = {}
    real = engine.EngineProcess.start

    def spy(self, cmd, env, log_path=None):
        seen["cmd"], seen["env"], seen["log"] = list(cmd), dict(env), log_path
        return real(self, cmd, env, log_path)

    monkeypatch.setattr(engine.EngineProcess, "start", spy)
    rtmp(ctrl)
    ctrl.start(KEY)
    assert wait_for(ctrl, {engine.PROTECTED}) == engine.PROTECTED
    assert not any(KEY in part for part in seen["cmd"]) and "--output" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--output") + 1] == f"env:{engine.OUTPUT_ENV}"
    assert seen["env"][engine.OUTPUT_ENV].endswith("/" + KEY)
    assert seen["env"]["BACKEND"] == "onnx" and seen["env"]["TRT"] == "0"
    time.sleep(0.3)
    assert lines and not any(KEY in line for line in lines)  # the engine printed the address; the window never shows it
    assert any("***" in line for line in lines)
    ctrl.stop()
    assert ctrl.status() == engine.STOPPED and ctrl.engine.exit_code == 0
    log = seen["log"].read_text(encoding="utf-8")  # the run is kept in a log file, also without the key
    assert seen["log"].parent.name == "logs" and "engine ended" in log and KEY not in log and "***" in log


def test_the_whole_picture_covered_is_shown_as_covered(ctrl, monkeypatch):
    monkeypatch.setenv("FAKE_FULL", "1")
    ctrl.start("")
    assert wait_for(ctrl, {engine.COVERED}) == engine.COVERED


def test_a_crash_is_failed_and_explained(ctrl, monkeypatch):
    monkeypatch.setenv("FAKE_CRASH", "1")
    ctrl.start("")
    assert wait_for(ctrl, {engine.FAILED}) == engine.FAILED
    assert ctrl.last_exit == 3 and ctrl.exit_reason() == "engine_exit"


def test_exit_reasons_by_code_and_text(ctrl):
    ctrl.last_exit = 4
    assert ctrl.exit_reason() == "engine_expired"
    ctrl.last_exit = 5
    assert ctrl.exit_reason() == "engine_no_heads"
    ctrl.last_exit, ctrl._engine_lines = 1, ["NO_ENCODER no H.264 encoder works"]
    assert ctrl.exit_reason() == "engine_no_encoder"


def test_nothing_starts_when_anything_is_unclear(ctrl, monkeypatch):
    rtmp(ctrl)
    assert ctrl.blockers(KEY) == []
    assert ctrl.blockers("") == ["bad_key"] and ctrl.blockers("a<b") == ["bad_key"]
    assert ctrl.blockers(" " + KEY + " ") == []  # the stray spaces of a paste are taken off
    ctrl.settings.rtmp_url = "http://nope"
    assert "bad_rtmp" in ctrl.blockers(KEY)
    rtmp(ctrl)
    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: True)
    assert ctrl.blockers(KEY) == ["why_expired"]
    with pytest.raises(RuntimeError, match="why_expired"):
        ctrl.start(KEY)
    assert not ctrl.engine.running  # an expired beta never streams, so it can never stream uncovered
    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: False)
    monkeypatch.setattr(modelpack, "problem", lambda models, quick=True: "missing")
    assert ctrl.blockers(KEY) == ["why_models"]
    monkeypatch.setattr(modelpack, "problem", lambda models, quick=True: "")
    monkeypatch.setattr(heads, "why_not", lambda env=None, app_folder=None: "placeholder")
    assert ctrl.blockers(KEY) == ["why_heads_placeholder"]
    monkeypatch.setattr(heads, "why_not", lambda env=None, app_folder=None: "missing")
    assert ctrl.blockers(KEY) == ["why_heads"]
    monkeypatch.setattr(heads, "why_not", lambda env=None, app_folder=None: "")
    ctrl.check_result = controller.CheckResult("fail")
    assert ctrl.blockers(KEY) == ["why_check"]


def test_a_running_engine_blocks_a_second_start(ctrl):
    ctrl.start("")
    wait_for(ctrl, {engine.PROTECTED})
    assert ctrl.blockers("") == ["why_running"]
    with pytest.raises(RuntimeError):
        ctrl.start("")


def test_old_state_file_is_not_mistaken_for_a_new_run(ctrl):
    state = paths.obs_state_file(ctrl.env)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("1 99999 0 0", encoding="ascii")
    ctrl.start("")
    assert ctrl.status() in (engine.STARTING, engine.PROTECTED)  # not PROTECTED from the leftover line alone


def test_remember_key_only_when_asked_and_protected(ctrl, tmp_path):
    ctrl.settings.remember_key = False
    assert ctrl.remember(KEY) and ctrl.stored_key() == "" and not paths.key_file(ctrl.env).exists()
    ctrl.settings.remember_key = True
    assert ctrl.remember(KEY) and ctrl.stored_key() == KEY
    assert KEY.encode() not in paths.key_file(ctrl.env).read_bytes()
    ctrl.settings.remember_key = False
    assert ctrl.remember(KEY) and not paths.key_file(ctrl.env).exists()  # unticking forgets it
    plain = controller.Controller({"STREAMSAFE_DATA_DIR": str(tmp_path / "d2")}, protector=None)
    plain.keys._protector = None  # a PC without a protected store
    plain.settings.remember_key = True
    assert plain.remember(KEY) is False and plain.settings.remember_key is False


def test_check_output_parsing_and_nvidia_decision(ctrl):
    out = ['{"verdict": "warn", "checks": [{"name": "gpu", "status": "ok", "detail": "NVIDIA GeForce RTX 3060: 12 GB"}, '
           '{"name": "engines", "status": "ok", "detail": "ready"}]}']  # fmt: skip
    r = controller.parse_check_output(["noise", *out])
    assert r.verdict == "warn" and r.item("gpu")["detail"].startswith("NVIDIA")
    ctrl.check_result = r
    assert ctrl.has_nvidia_gpu()
    assert controller.parse_check_output(["Traceback", "boom"]).error.endswith("boom")
    assert controller.parse_check_output([]).verdict == ""


def test_update_flow_reports_but_never_blocks(ctrl, monkeypatch):
    monkeypatch.setattr(updates, "check", lambda serial=0: (_ for _ in ()).throw(updates.UpdateError("unreachable")))
    assert ctrl.check_updates() is None and ctrl.update_wanted() == (False, False)
    monkeypatch.setattr(updates, "check", lambda serial=0: None)
    assert ctrl.check_updates() is None
    ctrl.update = updates.Update(serial=1, app_version="99.0.0")
    assert ctrl.update_wanted() == (True, False)


def test_streamer_logo_is_kept_as_a_copy_and_goes_on_the_full_screen_cover(ctrl, tmp_path, monkeypatch):
    import cv2
    import numpy as np

    src = tmp_path / "شعار.jpg"  # an Arabic file name, as on a real streamer's PC
    _, data = cv2.imencode(".jpg", np.full((2000, 3000, 3), 120, np.uint8))
    src.write_bytes(data.tobytes())
    assert ctrl.set_brand_image(str(src)) == ""
    kept = ctrl.settings.brand_image
    assert kept.endswith("logo.png") and "brand" in kept
    img = cv2.imdecode(np.fromfile(kept, np.uint8), cv2.IMREAD_UNCHANGED)
    assert max(img.shape[:2]) == ctrl.BRAND_MAX  # shrunk
    src.unlink()  # the original may move: the copy is used
    from streamsafe.app import paths

    ours = str(paths.bundled_brand_dir() / "logo.png")
    # the normal plan: StreamSafe's logo on the covers, whatever the streamer picked
    env = engine.engine_env({}, False, data_env=ctrl.env, brand_image=kept)
    assert env["SCENE_IMAGE"] == env["SLATE_IMAGE"] == ours and env["LOGO"].endswith("logo_rgba.png")
    # Pro: the streamer's own logo on the card and on the clothing
    env = engine.engine_env({}, False, data_env={**ctrl.env, "STREAMSAFE_PLAN": "pro"}, brand_image=kept)
    assert env["SCENE_IMAGE"] == env["SLATE_IMAGE"] == env["LOGO"] == kept and env["SCENE_TEXT"] == env["SLATE_TEXT"] == ""
    bad = tmp_path / "not_an_image.png"
    bad.write_text("hello")
    assert ctrl.set_brand_image(str(bad)) == "brand_bad" and ctrl.settings.brand_image == kept
    assert ctrl.set_brand_image("") == "" and ctrl.settings.brand_image == ""
    assert engine.engine_env({"SCENE_IMAGE": "old"}, False, data_env=ctrl.env)["SCENE_IMAGE"] == ours  # never a stale value
