"""Product runtime settings: backend, folders, TensorRT only when ready, DirectML otherwise, stream key handling."""

import json
import sys
from pathlib import Path

import pytest

from streamsafe.app import encoders, engine, entry, keystore, nvidia_pack, paths, release, runtime_select
from streamsafe.app import settings as cfg
from tests.appfix import FakeProtector, make_heads


def env_for(tmp_path, **extra):
    return {"STREAMSAFE_DATA_DIR": str(tmp_path / "data"), "PATH": "", **extra}


def fake_nvidia_ready(tmp_path):
    site = nvidia_pack.site_dir(paths.nvidia_dir(env_for(tmp_path)))
    (site / "onnxruntime").mkdir(parents=True)
    (site / "onnxruntime" / "__init__.py").write_text("")
    (site / nvidia_pack.READY).write_text(json.dumps({"wheels": {w.name: w.sha256 for w in release.NVIDIA_WHEELS}}))
    return site


# --- product defaults -----------------------------------------------------------------------------------------
def test_defaults_are_the_onnx_product_path_on_directml(tmp_path):
    env = env_for(tmp_path, LOCALAPPDATA=str(tmp_path / "ignored"))
    entry.product_defaults(env)
    assert env["BACKEND"] == "onnx"
    assert Path(env["ONNX_DIR"]) == tmp_path / "data" / "models"
    assert Path(env["OBS_STATE"]).parent == tmp_path / "data" / "obs"
    assert "TRT" not in env and runtime_select.RUNTIME_ENV not in env  # DirectML: no TensorRT


def test_default_data_dir_is_under_localappdata_never_programdata(tmp_path):
    env = {"LOCALAPPDATA": str(tmp_path / "AppData" / "Local")}
    assert paths.data_dir(env) == tmp_path / "AppData" / "Local" / "StreamSafe"
    for f in (paths.models_dir, paths.obs_state_file, paths.reports_dir, paths.key_file, paths.settings_file, paths.nvidia_dir):
        assert "ProgramData" not in str(f(env)) and str(f(env)).startswith(str(tmp_path / "AppData"))


def test_trt_only_when_nvidia_runtime_is_ready(tmp_path):
    env = env_for(tmp_path)
    entry.product_defaults(env)
    assert "TRT" not in env
    fake_nvidia_ready(tmp_path)
    env = env_for(tmp_path)
    entry.product_defaults(env)
    assert env[runtime_select.RUNTIME_ENV] == "nvidia" and env["TRT"] == "1" and env["TRT_BUILD"] == "0"
    stale = env_for(tmp_path)
    (nvidia_pack.site_dir(paths.nvidia_dir(stale)) / nvidia_pack.READY).write_text(json.dumps({"wheels": {"old": "x"}}))
    entry.product_defaults(stale)  # libraries of another build: not ready, DirectML
    assert "TRT" not in stale


def test_environment_wins_over_defaults(tmp_path):
    env = env_for(tmp_path, BACKEND="torch", ONNX_DIR="/x", HEAD_DIR="/h")
    entry.product_defaults(env)
    assert (env["BACKEND"], env["ONNX_DIR"], env["HEAD_DIR"]) == ("torch", "/x", "/h")
    env2 = env_for(tmp_path)
    entry.product_defaults(env2, models="/picked")
    assert env2["ONNX_DIR"] == "/picked"


def test_head_dir_is_the_best_usable_heads_folder(tmp_path, monkeypatch):
    app = tmp_path / "app"
    monkeypatch.setattr(paths, "app_dir", lambda: app)
    env = env_for(tmp_path)
    make_heads(app / "heads", version=1)
    entry.product_defaults(env)
    assert Path(env["HEAD_DIR"]) == app / "heads"
    env = env_for(tmp_path)
    make_heads(paths.heads_dir(env), version=2)
    e2 = dict(env)
    entry.product_defaults(e2)
    assert Path(e2["HEAD_DIR"]) == paths.heads_dir(env)


def test_engine_env_directml_and_nvidia(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "app_dir", lambda: tmp_path / "app")
    where = env_for(tmp_path)
    plain = engine.engine_env({"PATH": "p", "TRT": "1"}, False, "rtmps://live.example/app/KEY1234", where)
    assert plain["BACKEND"] == "onnx" and plain["TRT"] == "0" and plain[runtime_select.RUNTIME_ENV] == "directml"
    assert plain["ONNX_DIR"] == str(tmp_path / "data" / "models") and plain[engine.PRODUCT_ENV] == "1"
    assert plain[engine.OUTPUT_ENV] == "rtmps://live.example/app/KEY1234"
    nv = engine.engine_env({"PATH": "p"}, True, "", where)
    assert nv["TRT"] == "1" and nv["TRT_BUILD"] == "0" and nv[runtime_select.RUNTIME_ENV] == "nvidia" and engine.OUTPUT_ENV not in nv


def test_runtime_select_only_points_at_a_sealed_folder(tmp_path):
    env = env_for(tmp_path)
    path: list[str] = []
    assert runtime_select.activate(env, path) == "not asked"
    env[runtime_select.RUNTIME_ENV] = "nvidia"
    if tuple(sys.version_info[:2]) != release.NVIDIA_PYTHON:
        assert "needs Python" in runtime_select.activate(env, path)
        return
    assert "not installed" in runtime_select.activate(env, path) and path == []
    site = fake_nvidia_ready(tmp_path)
    assert runtime_select.activate(env, path) == "" and path[0] == str(site)
    assert runtime_select.activate(env, path) == ""  # idempotent


# --- stream key and command lines ---------------------------------------------------------------------------
def test_stream_key_is_never_on_a_command_line():
    s = cfg.Settings(output=cfg.OUTPUT_RTMP, rtmp_url="rtmps://live.example/app", input=cfg.INPUT_SRT)
    args = engine.run_args(s, Path("reports"))
    assert args[args.index("--output") + 1] == f"env:{engine.OUTPUT_ENV}"
    assert not any("rtmps://" in a for a in args)


def test_env_output_resolution_and_missing_variable():
    args = ["run", "--input", "x", "--output", "env:SS_OUT"]
    assert entry.resolve_env_outputs(args, {"SS_OUT": "rtmps://a/b/KEY"})[-1] == "rtmps://a/b/KEY"
    assert entry.resolve_env_outputs(["--output=env:SS_OUT"], {"SS_OUT": "u"}) == ["--output=u"]
    with pytest.raises(SystemExit, match="not set"):
        entry.resolve_env_outputs(args, {})
    assert entry.resolve_env_outputs(["--output", "window"], {}) == ["--output", "window"]


def test_redactor_blanks_key_and_url():
    r = engine.Redactor("rtmps://live.example/app/SECRETKEY", "SECRETKEY", "ab")
    assert r.clean("failed to open rtmps://live.example/app/SECRETKEY: x SECRETKEY") == "failed to open ***: x ***"
    assert r.clean("ab stays") == "ab stays"  # shorter than 4 characters: not treated as a secret


# --- keystore ------------------------------------------------------------------------------------------------
def test_keystore_roundtrip_is_not_plain_text(tmp_path):
    ks = keystore.KeyStore(tmp_path / "k.bin", FakeProtector())
    assert ks.available and ks.load() == ""
    ks.save("live_123_secret")
    raw = (tmp_path / "k.bin").read_bytes()
    assert raw.startswith(keystore.MAGIC) and b"live_123_secret" not in raw
    assert ks.load() == "live_123_secret"
    ks.clear()
    assert ks.load() == "" and not (tmp_path / "k.bin").exists()


def test_keystore_without_a_protector_stores_nothing(tmp_path):
    ks = keystore.KeyStore(tmp_path / "k.bin", None)
    assert not ks.available
    with pytest.raises(keystore.NoSecureStoreError):
        ks.save("abc")
    assert not (tmp_path / "k.bin").exists() and ks.load() == ""


def test_keystore_refuses_bad_input_and_damaged_files(tmp_path):
    ks = keystore.KeyStore(tmp_path / "k.bin", FakeProtector())
    for bad in ("", "x" * 600):
        with pytest.raises(ValueError):
            ks.save(bad)
    (tmp_path / "k.bin").write_bytes(b"garbage")
    assert ks.load() == ""
    (tmp_path / "k.bin").write_bytes(keystore.MAGIC + b"\xff\xfe\xfd")
    assert ks.load() == ""  # cannot be decoded: "no key", never an exception


def test_default_protector_exists_only_on_windows():
    p = keystore.default_protector()
    assert (p is not None) == (sys.platform == "win32")


# --- settings ---------------------------------------------------------------------------------------------------
def test_settings_never_hold_the_key_and_survive_garbage(tmp_path):
    f = tmp_path / "s.json"
    cfg.save(f, cfg.Settings(lang="en", rtmp_url="rtmps://x/y", srt_port=9100))
    text = f.read_text(encoding="utf-8")
    assert "key" not in text.replace("remember_key", "")
    assert cfg.load(f).lang == "en" and cfg.load(f).srt_port == 9100
    f.write_text('{"lang": 5, "srt_port": "x", "input": "nope", "delay": 3}', encoding="utf-8")
    s = cfg.load(f)
    assert s.lang == "ar" and s.srt_port == 9000 and s.input == cfg.INPUT_CAMERA and s.delay == 3.0
    f.write_text("not json", encoding="utf-8")
    assert cfg.load(f) == cfg.Settings()


def test_settings_problems_and_validators():
    s = cfg.Settings(input=cfg.INPUT_SRT, srt_port=80, output=cfg.OUTPUT_RTMP, rtmp_url="http://x", delay=99)
    assert s.problems() == ["bad_port", "bad_rtmp", "bad_delay"]
    assert cfg.Settings().problems() == []
    assert cfg.valid_rtmp_url("rtmps://live.twitch.tv/app") and not cfg.valid_rtmp_url("rtmp://") and not cfg.valid_rtmp_url("rtmp://a b")
    assert cfg.valid_key("live_1234_abc") and not cfg.valid_key("a b") and not cfg.valid_key("") and not cfg.valid_key("a/b")


def test_obs_state_file_is_per_user_and_ascii():
    arabic = {"LOCALAPPDATA": "C:\\Users\\علي\\AppData\\Local"}
    plain = paths.obs_state_file({"LOCALAPPDATA": "C:\\Users\\ali\\AppData\\Local"}, windows=True)
    assert str(plain).isascii() and "ProgramData" not in str(plain)
    assert paths.obs_state_file(arabic, windows=False).parts[-1] == "state.txt"
    short = paths.obs_state_file(arabic, windows=True, short_path=lambda p: "C:\\Users\\E0F1~1\\AppData\\Local\\StreamSafe\\obs")
    assert str(short).isascii() and "E0F1~1" in str(short)
    fallback = paths.obs_state_file({**arabic, "PUBLIC": "C:\\Users\\Public"}, windows=True, short_path=lambda p: "")
    assert str(fallback).isascii() and "ProgramData" not in str(fallback) and "Public" in str(fallback)


# --- status words ------------------------------------------------------------------------------------------------
def test_status_is_protected_only_on_positive_evidence():
    now = [0.0]
    t = engine.StatusTracker(lambda: now[0])
    alive = lambda seq, full=False: engine.State(True, seq, False, full)  # noqa: E731
    assert t.update(False, None) == engine.STOPPED and t.update(False, None, failed=True) == engine.FAILED
    assert t.update(True, None) == engine.STARTING
    assert t.update(True, alive(5)) == engine.STARTING  # first sight of a counter: maybe left over from an earlier run
    now[0] += 0.3
    assert t.update(True, alive(6)) == engine.PROTECTED
    now[0] += 0.3
    assert t.update(True, alive(7, full=True)) == engine.COVERED
    now[0] += 0.3
    assert t.update(True, alive(8)) == engine.PROTECTED
    now[0] += 2.0
    assert t.update(True, alive(8)) == engine.STALLED  # the counter stopped: never "protected"
    assert t.update(True, engine.State(False, 9, False, False)) in (engine.STALLED, engine.STARTING)


def test_state_file_parsing(tmp_path):
    f = tmp_path / "s.txt"
    assert engine.read_state(f) is None
    f.write_text("1 42 0 1", encoding="ascii")
    assert engine.read_state(f) == engine.State(True, 42, False, True)
    f.write_text("garbage", encoding="ascii")
    assert engine.read_state(f) is None


# --- encoders (LGPL FFmpeg has no libx264) --------------------------------------------------------------------
class Result:
    def __init__(self, code=0, out=""):
        self.returncode, self.stdout = code, out


def fake_run(listed, working):
    def run(cmd, **kw):
        if "-encoders" in cmd:
            return Result(0, "\n".join(f" V....D {e}  encoder" for e in listed))
        enc = cmd[cmd.index("-c:v") + 1]
        return Result(0 if enc in working else 1)

    return run


def test_encoder_choice_tries_each_and_takes_the_first_that_works():
    run = fake_run(["h264_nvenc", "h264_mf", "libx264"], working={"h264_mf"})
    assert encoders.choose("ffmpeg", run) == "h264_mf"  # nvenc is listed but does not work on this PC
    assert encoders.choose("ffmpeg", fake_run(["h264_nvenc"], set())) == ""
    assert encoders.choose("ffmpeg", fake_run(["h264_nvenc", "h264_amf"], {"h264_nvenc", "h264_amf"})) == "h264_nvenc"


def test_choose_into_only_for_network_streams_and_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "find_ffmpeg", lambda env=None, root=None: "ffmpeg")
    env: dict[str, str] = {}
    encoders.choose_into(env, ["run", "--output", "window"], fake_run([], set()))
    assert "ENCODER" not in env  # a window output needs no encoder
    encoders.choose_into(env, ["run", "--output", "rtmps://x/y"], fake_run(["h264_mf"], {"h264_mf"}))
    assert env["ENCODER"] == "h264_mf"
    env2: dict[str, str] = {}
    with pytest.raises(SystemExit, match="NO_ENCODER"):
        encoders.choose_into(env2, ["run", "--output=rtmp://x/y"], fake_run(["h264_nvenc"], set()))
    env3 = {"ENCODER": "libx264"}
    encoders.choose_into(env3, ["run", "--output", "rtmp://x/y"], fake_run([], set()))
    assert env3["ENCODER"] == "libx264"  # an explicit choice is kept


def test_sink_encoder_default_follows_the_environment(monkeypatch):
    """The encoder is read from $ENCODER when the sink is made (the app sets it after its probe), else libx264."""
    from streamsafe.io import sinks

    s = sinks.FFmpegSink("out.ts", (64, 36), 30.0, spawn=lambda cmd, **kw: None)
    monkeypatch.setenv("ENCODER", "h264_mf")
    assert "h264_mf" in sinks.FFmpegSink("out.ts", (64, 36), 30.0, spawn=lambda cmd, **kw: None)._cmd
    assert "h264_nvenc" in sinks.FFmpegSink("out.ts", (64, 36), 30.0, encoder="h264_nvenc", spawn=lambda cmd, **kw: None)._cmd
    monkeypatch.delenv("ENCODER")
    assert "libx264" in sinks.FFmpegSink("out.ts", (64, 36), 30.0, spawn=lambda cmd, **kw: None)._cmd
    assert "libx264" in s._cmd
