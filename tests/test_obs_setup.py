"""StreamSafe sets OBS up by itself: only the server (our address, no key); the streamer's own setup is kept; OBS open -> untouched."""

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
    assert "VBitrate=2500" in (prof / "basic.ini").read_text(encoding="utf-8-sig")  # their video/bitrate untouched
    assert not (prof / "basic.ini.before-streamsafe").exists()
    assert "live_secret" in (prof / "service.json.before-streamsafe").read_text(encoding="utf-8")  # their old setup kept
    assert obs_setup.is_set(SERVER, env)
    assert obs_setup.apply(SERVER, env, running=lambda: False) == ""  # again: same result, first copy kept
    assert "live_secret" in (prof / "service.json.before-streamsafe").read_text(encoding="utf-8")


def scene(tmp_path, sources):
    f = tmp_path / "obs-studio" / "basic" / "scenes" / "Untitled.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    data = {"name": "Untitled", "current_scene": "Scene", "scene_order": [{"name": "Scene"}], "sources": sources}
    f.write_text(json.dumps(data), "utf-8")
    return f


def test_empty_obs_gets_a_picture_source_and_a_real_setup_is_never_touched(tmp_path):
    env = obs_folder(tmp_path)
    (tmp_path / "obs-studio" / "user.ini").write_text("[Basic]\nProfileDir=Untitled\nSceneCollectionFile=Untitled\n", "utf-8")
    f = scene(tmp_path, [{"id": "scene", "name": "Scene", "settings": {"items": []}}])
    assert obs_setup.apply(SERVER, env, running=lambda: False) == ""
    data = json.loads(f.read_text("utf-8"))
    kinds = [s["id"] for s in data["sources"]]
    assert kinds == ["scene", "monitor_capture", "game_capture"]
    items = data["sources"][0]["settings"]["items"]
    assert [i["name"] for i in items] == ["Display capture", "Game capture"]  # game on top
    assert {i["source_uuid"] for i in items} == {s["uuid"] for s in data["sources"][1:]}
    assert obs_setup.apply(SERVER, env, running=lambda: False) == ""  # again: nothing added twice
    assert [s["id"] for s in json.loads(f.read_text("utf-8"))["sources"]] == kinds

    mine = [{"id": "scene", "name": "Scene", "settings": {"items": [{"name": "Cam"}]}}, {"id": "dshow_input", "name": "Cam"}]
    f = scene(tmp_path, mine)
    obs_setup.apply(SERVER, env, running=lambda: False)
    assert json.loads(f.read_text("utf-8"))["sources"] == mine


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


def test_our_picture_follows_obs_own_video_settings(tmp_path):
    from streamsafe.app import engine
    from streamsafe.app import settings as cfg

    env = obs_folder(tmp_path)  # BaseCX only: OBS's default output
    prof = tmp_path / "obs-studio" / "basic" / "profiles" / "Untitled" / "basic.ini"
    prof.write_text("[Video]\nOutputCX=2560\nOutputCY=1440\nFPSType=1\nFPSInt=60\n", "utf-8")
    assert obs_setup.obs_video(env) == (2560, 1440, 60.0)
    s = cfg.Settings(resolution="1080p", fps=30)
    engine.follow_obs(s, obs_setup.obs_video(env))
    assert (s.resolution, s.fps) == ("1440p", 30)  # live is always 30 fps (60 broke up on mid-range PCs)
    prof.write_text("[Video]\nOutputCX=1600\nOutputCY=900\nFPSCommon=29.97\n", "utf-8")
    engine.follow_obs(s, obs_setup.obs_video(env))
    assert (s.resolution, s.fps) == ("1080p", 30)  # never smaller than what OBS sends
    s.follow_obs = False  # the streamer chose by hand in Advanced
    engine.follow_obs(s, (3840, 2160, 60.0))
    assert (s.resolution, s.fps) == ("1080p", 30)
    assert obs_setup.obs_video({"APPDATA": str(tmp_path / "none")}) is None
