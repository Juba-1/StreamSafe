"""StreamSafe sets OBS up by itself: Custom server = our address, no key, a high local bitrate; OBS open -> untouched."""

import json
from pathlib import Path

from streamsafe.app import obs_setup

SERVER = "srt://127.0.0.1:9000"


def obs_folder(tmp_path: Path, mode: str = "Simple", ini_name: str = "user.ini", profile: str = "Untitled") -> dict:
    root = tmp_path / "obs-studio"
    prof = root / "basic" / "profiles" / profile
    prof.mkdir(parents=True)
    (root / ini_name).write_text(f"﻿[General]\nFirstRun=true\n\n[Basic]\nProfile={profile}\nProfileDir={profile}\n", encoding="utf-8")
    (prof / "basic.ini").write_text(
        f"﻿[General]\nName={profile}\n\n[Output]\nMode={mode}\n\n[SimpleOutput]\nVBitrate=2500\nABitrate=160\nRecEncoder=x264\n\n[Video]\nBaseCX=1920\n",
        encoding="utf-8",
    )
    old = {"type": "rtmp_common", "settings": {"service": "Twitch", "key": "live_secret"}}
    (prof / "service.json").write_text(json.dumps(old), encoding="utf-8")
    return {"APPDATA": str(tmp_path)}


def test_points_obs_at_streamsafe_and_keeps_a_copy(tmp_path):
    env = obs_folder(tmp_path)
    assert not obs_setup.is_set(SERVER, env)
    assert obs_setup.apply(SERVER, env, running=lambda: False) == ""
    prof = tmp_path / "obs-studio" / "basic" / "profiles" / "Untitled"
    svc = json.loads((prof / "service.json").read_text(encoding="utf-8"))
    assert svc == {"type": "rtmp_custom", "settings": {"server": SERVER, "key": "", "use_auth": False, "bwtest": False}}
    ini = (prof / "basic.ini").read_text(encoding="utf-8-sig")
    assert "VBitrate=20000" in ini and "RecEncoder=x264" in ini and "BaseCX=1920" in ini  # the rest kept, case kept
    assert "live_secret" in (prof / "service.json.before-streamsafe").read_text(encoding="utf-8")  # their old setup kept
    assert obs_setup.is_set(SERVER, env)
    assert obs_setup.apply(SERVER, env, running=lambda: False) == ""  # again: same result, first copy kept
    assert "live_secret" in (prof / "service.json.before-streamsafe").read_text(encoding="utf-8")


def test_advanced_output_mode_uses_the_stream_encoder(tmp_path):
    env = obs_folder(tmp_path, mode="Advanced", ini_name="global.ini", profile="Main")
    prof = tmp_path / "obs-studio" / "basic" / "profiles" / "Main"
    (prof / "streamEncoder.json").write_text(json.dumps({"bitrate": 6000, "preset": "p5"}), encoding="utf-8")
    assert obs_setup.apply(SERVER, env, running=lambda: False) == ""
    enc = json.loads((prof / "streamEncoder.json").read_text(encoding="utf-8"))
    assert enc == {"bitrate": 20000, "preset": "p5", "rate_control": "CBR", "keyint_sec": 2}


def test_obs_open_is_left_alone(tmp_path):
    env = obs_folder(tmp_path)
    assert obs_setup.apply(SERVER, env, running=lambda: True) == "running"
    assert not obs_setup.is_set(SERVER, env)


def test_no_obs(tmp_path):
    assert obs_setup.apply(SERVER, {"APPDATA": str(tmp_path)}, running=lambda: False) == "no_obs"
    assert not obs_setup.is_set(SERVER, {"APPDATA": str(tmp_path)})


def test_launch_starts_streaming_from_obs_folder(tmp_path):
    seen = {}
    exe = tmp_path / "bin" / "obs64.exe"
    assert obs_setup.launch(exe, popen=lambda args, **kw: seen.update(args=args, **kw))
    assert seen["args"] == [str(exe), "--startstreaming"] and seen["cwd"] == str(exe.parent)
