"""The streamer's protection choices: whole-screen mode, the panic key, the trial without going live, the
pre-stream warnings and putting OBS back."""

from __future__ import annotations

import json

import numpy as np

from streamsafe.app import hotkey, obs_setup, preflight
from streamsafe.config import RenderConfig
from streamsafe.runtime.gate import OutputGate
from streamsafe.types import Decision, Layer

W, H = 1920, 1080


def frame() -> np.ndarray:
    return np.full((H, W, 3), 100, np.uint8)


def layer() -> Layer:
    mask = np.ones((100, 100), bool)
    return Layer(tid=0, y=500, x=900, mask=mask, fabric=np.full((100, 100, 3), 7, np.uint8), centroid=(550.0, 950.0))


def test_screen_mode_covers_the_whole_screen_and_holds_it(monkeypatch):
    monkeypatch.setenv("COVER_MODE", "screen")
    gate = OutputGate(RenderConfig(motion=False), (W, H), fps=10)
    gate.add_decision(Decision(frame_id=0, ok=True, layers=[layer()]))
    for i in range(1, 30):
        gate.add_decision(Decision(frame_id=i, ok=True))
    assert gate.render(0, frame()) is gate.card
    assert gate.render(10, frame()) is gate.card, "held 1.5 s after the last person who needed covering"
    assert gate.render(20, frame()) is not gate.card


def test_people_mode_covers_only_the_person(monkeypatch):
    monkeypatch.delenv("COVER_MODE", raising=False)
    gate = OutputGate(RenderConfig(motion=False), (W, H))
    gate.add_decision(Decision(frame_id=0, ok=True, layers=[layer()]))
    assert gate.render(0, frame()) is not gate.card


def test_panic_covers_at_once_and_hands_back():
    gate = OutputGate(RenderConfig(motion=False), (W, H))
    for i in range(3):
        gate.add_decision(Decision(frame_id=i, ok=True))
    gate.manual = True
    assert gate.render(0, frame()) is gate.card
    gate.manual = False
    assert gate.render(2, frame()) is not gate.card


def test_hotkey_names():
    assert hotkey.parse("F9") == (0, 0x78)
    assert hotkey.parse("Ctrl+F9") == (0x0002, 0x78)
    assert hotkey.parse("Banana") is None
    assert all(hotkey.parse(k) for k in hotkey.CHOICES)


def test_controller_preview_and_panic(tmp_path, monkeypatch):
    from streamsafe.app import controller

    ctrl = controller.Controller(env={"LOCALAPPDATA": str(tmp_path), "APPDATA": str(tmp_path)})
    ctrl.set_panic(True)
    assert ctrl.panic
    ctrl.set_panic(False)
    assert not ctrl.panic
    from dataclasses import replace

    from streamsafe.app.settings import OUTPUT_WINDOW

    no_key = ctrl.blockers("", replace(ctrl.settings, output=OUTPUT_WINDOW))
    assert "bad_key" not in no_key, "the trial needs no stream key"


def _obs(tmp_path):
    root = tmp_path / "obs-studio"
    prof = root / "basic" / "profiles" / "P"
    prof.mkdir(parents=True)
    (root / "user.ini").write_text("[Basic]\nProfileDir=P\n", encoding="utf-8")
    return {"APPDATA": str(tmp_path)}, root, prof


def test_restore_puts_obs_back(tmp_path):
    env, _, prof = _obs(tmp_path)
    own = {"type": "rtmp_common", "settings": {"service": "Twitch", "key": "mine"}}
    (prof / "service.json").write_text(json.dumps(own), encoding="utf-8")
    assert obs_setup.apply("srt://127.0.0.1:9000", env, running=lambda: False) == ""
    assert obs_setup.restore(env, running=lambda: True) == "running"
    assert obs_setup.restore(env, running=lambda: False) == ""
    assert json.loads((prof / "service.json").read_text(encoding="utf-8")) == own
    assert obs_setup.restore(env, running=lambda: False) == "no_backup"


def test_preflight_warnings(tmp_path):
    env, root, prof = _obs(tmp_path)
    assert preflight.warnings(env) == []
    (prof / "basic.ini").write_text("[Output]\nDelayEnable=true\nDelaySec=5\n", encoding="utf-8")
    (root / "plugin_config" / "aitum-multistream").mkdir(parents=True)
    assert set(preflight.warnings(env)) == {"pre_obs_delay", "pre_multistream"}
    err = OSError("[WinError 4551] An Application Control policy has blocked this file")
    assert preflight.blocked_by_windows(err)


# ---------------------------------------------------------------------------------- a second platform
def test_second_platform_args_env_and_status(tmp_path):
    from streamsafe.app import engine
    from streamsafe.app.entry import resolve_env_outputs
    from streamsafe.app.settings import Settings

    s = Settings(platform="kick", rtmp_url="rtmps://a.example/app", platform2="twitch", rtmp_url2="rtmp://b.example/app")
    args = engine.run_args(s, tmp_path)
    assert args[args.index("--extra-output") + 1] == f"env:{engine.OUTPUT2_ENV}"
    env = engine.engine_env({}, False, "rtmps://a/app/k1", {"LOCALAPPDATA": str(tmp_path)}, url2="rtmp://b/app/k2")
    assert env[engine.OUTPUT2_ENV] == "rtmp://b/app/k2"
    assert resolve_env_outputs(["--extra-output", f"env:{engine.OUTPUT2_ENV}"], env)[1] == "rtmp://b/app/k2"
    lk = engine.LinkState()
    lk.take({"event": "platform_connected"})
    lk.take({"event": "platform_failed", "out": "2", "reason": "auth"})
    assert (lk.platform, lk.platform2) == ("connected", "failed"), "the extra platform never changes the main one"
    assert "bad_rtmp2" in Settings(platform="kick", rtmp_url="rtmps://a.example/app", platform2="custom", rtmp_url2="x").problems()


def test_best_effort_drops_a_failing_extra():
    from streamsafe.io.sinks import BestEffort, MultiSink

    class Good:
        frames = 0

        def write(self, frame):
            Good.frames += 1

        def close(self):
            pass

    class Bad(Good):
        def write(self, frame):
            raise ConnectionError("refused")

    multi = MultiSink([Good(), BestEffort(Bad(), "2")])
    for _ in range(3):
        multi.write(frame())
    assert Good.frames == 3, "the main platform keeps every frame"


def test_vertical_obs_canvas_keeps_its_shape(tmp_path):
    from streamsafe.app import engine
    from streamsafe.app.settings import Settings

    s = Settings(platform="tiktok")
    engine.follow_obs(s, (1080, 1920, 30.0))
    args = engine.run_args(s, tmp_path)
    assert args[args.index("--size") + 1] == "1080x1920", "a TikTok vertical canvas is never squeezed into 16:9"
    s = Settings(platform="kick")
    engine.follow_obs(s, (2560, 1440, 60.0))
    assert engine.run_args(s, tmp_path)[engine.run_args(s, tmp_path).index("--size") + 1] == "1920x1080"


# ---------------------------------------------------------------------------------- live control
def test_live_control_file(tmp_path):
    import os
    import time as _t

    from streamsafe.runtime.control import LiveControl, parse

    assert parse({"zones": [[0.7, 0.7, 1.2, 1.0], [0.5, 0.5, 0.4, 0.6], "x"], "thr_scale": 9}).zones == ((0.7, 0.7, 1.0, 1.0),)
    f = tmp_path / "control.json"
    live = LiveControl(f, every_s=0)
    assert live.get().panic is False, "no file: nothing changes"
    f.write_text('{"panic": true, "thr_scale": 1.8, "zones": [[0.75, 0.0, 1.0, 0.3]]}', encoding="utf-8")
    got = live.get()
    assert got.panic and got.thr_scale == 1.8 and got.in_zone(0.9, 0.1) and not got.in_zone(0.5, 0.5)
    f.write_text("{half", encoding="utf-8")
    os.utime(f, (_t.time() + 5, _t.time() + 5))
    assert live.get().panic, "a half-written file keeps the last good value"


def test_webcam_zone_is_never_dressed():
    from streamsafe.runtime.control import Live

    gate = OutputGate(RenderConfig(motion=False), (W, H))
    gate.zones = Live(zones=((0.4, 0.4, 0.6, 0.6),))  # the layer's centre (950, 550) is inside
    d = Decision(frame_id=0, ok=True, layers=[layer()])
    gate.add_decision(d)
    assert d.layers == []


def test_controller_writes_the_live_control(tmp_path):
    import json

    from streamsafe.app import controller

    ctrl = controller.Controller(env={"LOCALAPPDATA": str(tmp_path), "APPDATA": str(tmp_path)})
    ctrl.set_sensitivity(1.9)
    ctrl.set_zones([[0.7, 0.0, 1.0, 0.3], [2, 2, 1, 1]])
    ctrl.set_panic(True)
    data = json.loads(ctrl.control_file().read_text(encoding="utf-8"))
    assert data == {"panic": True, "everyone": False, "thr_scale": 1.9, "zones": [[0.7, 0.0, 1.0, 0.3]], "covers": [], "vcrop": None, "allow": []}
