"""Set OBS up for StreamSafe by itself, so the streamer does not touch OBS's settings.

Only what StreamSafe needs, nothing of the streamer's own setup (a copy is kept once as ``*.before-streamsafe``):
- Settings > Stream: Custom server = StreamSafe's local address, no key (``service.json``). Video size, frame rate,
  encoder, bitrate, audio and scenes stay as the streamer set them; StreamSafe follows what OBS sends.
- An OBS with no picture source at all (a fresh one) gets "Display capture" + "Game capture" in its scene, so the
  stream is not black. A scene that has any source is never touched.

OBS keeps its settings in memory and writes them back when it closes, so the files are only changed while OBS is
closed. Then StreamSafe opens OBS with ``--startstreaming``: OBS connects to StreamSafe without a click.
"""

from __future__ import annotations

import configparser
import contextlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

OBS_EXE = "obs64.exe"
OBS_PATHS = (r"C:\Program Files\obs-studio\bin\64bit\obs64.exe", r"C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe")
BACKUP = ".before-streamsafe"


def config_dir(env: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if env is None else env
    appdata = env.get("APPDATA", "")
    folder = Path(appdata) / "obs-studio" if appdata else None
    return folder if folder is not None and folder.is_dir() else None


def _ini(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str  # type: ignore[assignment,method-assign]  # OBS's keys are case sensitive
    with contextlib.suppress(OSError, configparser.Error):
        cp.read_string(path.read_text(encoding="utf-8-sig"))
    return cp


def profile_dir(root: Path) -> Path | None:
    """The profile OBS opens with (user.ini in OBS 31+, global.ini before), else the only/first one."""
    profiles = root / "basic" / "profiles"
    for name in ("user.ini", "global.ini"):
        cp = _ini(root / name)
        d = cp.get("Basic", "ProfileDir", fallback="")
        if d and (profiles / d).is_dir():
            return profiles / d
    found = sorted(p for p in profiles.glob("*") if p.is_dir()) if profiles.is_dir() else []
    return found[0] if found else None


def find_obs(env: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if env is None else env
    for p in (*OBS_PATHS, env.get("STREAMSAFE_OBS_EXE", ""), *_registered_obs(), *_other_obs_places(env)):
        if p and Path(p).is_file():
            return Path(p)
    which = shutil.which(OBS_EXE)
    return Path(which) if which else None


def _registered_obs() -> list[str]:
    """Where Windows says OBS is installed (its installer writes this key): finds installs on another drive."""
    if sys.platform != "win32":
        return []
    import winreg

    found = []
    for hive, key in ((winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\OBS Studio"), (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\OBS Studio")):
        try:
            with winreg.OpenKey(hive, key) as k:
                folder = winreg.QueryValueEx(k, "")[0]
        except OSError:
            continue
        found.append(str(Path(folder) / "bin" / "64bit" / OBS_EXE))
    return found


def _other_obs_places(env: Mapping[str, str]) -> list[str]:
    """The Steam copy of OBS and the usual folders on other drives."""
    bin64 = ("bin", "64bit", OBS_EXE)
    steam = ("steamapps", "common", "OBS Studio", *bin64)
    out = [str(Path(env[v], "Steam", *steam)) for v in ("ProgramFiles(x86)", "ProgramFiles") if env.get(v)]
    for drive in "CDEFG":
        root = drive + ":\\"
        for folder in ("obs-studio", "OBS Studio"):
            out += [str(Path(root, "Program Files", folder, *bin64)), str(Path(root, folder, *bin64))]
        out.append(str(Path(root, "SteamLibrary", *steam)))
    return out


def obs_running(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> bool:
    if sys.platform != "win32":
        return False
    try:
        out = run(
            ["tasklist", "/FI", f"IMAGENAME eq {OBS_EXE}", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return OBS_EXE.lower() in (out or "").lower()


def is_set(server: str, env: Mapping[str, str] | None = None) -> bool:
    """OBS already sends to StreamSafe."""
    root = config_dir(env)
    prof = profile_dir(root) if root else None
    if prof is None:
        return False
    try:
        d = json.loads((prof / "service.json").read_text(encoding="utf-8-sig"))
        return d.get("type") == "rtmp_custom" and d.get("settings", {}).get("server") == server
    except (OSError, ValueError, AttributeError):
        return False


def _backup(path: Path) -> None:
    keep = path.with_name(path.name + BACKUP)
    if path.is_file() and not keep.exists():
        shutil.copy2(path, keep)


def _write_json(path: Path, data: dict) -> None:
    _backup(path)
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def obs_kbps(env: Mapping[str, str] | None = None) -> int:
    """The video bitrate the streamer set in OBS (Settings > Output), 0 if unknown: what their internet carries."""
    root = config_dir(env)
    prof = profile_dir(root) if root else None
    if prof is None:
        return 0
    cp = _ini(prof / "basic.ini")
    try:
        if cp.get("Output", "Mode", fallback="Simple").lower() == "advanced":
            return int(json.loads((prof / "streamEncoder.json").read_text(encoding="utf-8")).get("bitrate", 0))
        return cp.getint("SimpleOutput", "VBitrate", fallback=0)
    except (OSError, ValueError, AttributeError):
        return 0


def obs_video(env: Mapping[str, str] | None = None) -> tuple[int, int, float] | None:
    """What OBS sends: its output width, height and frame rate (Settings > Video), or None if unknown."""
    root = config_dir(env)
    prof = profile_dir(root) if root else None
    if prof is None:
        return None
    cp = _ini(prof / "basic.ini")
    try:
        w = cp.getint("Video", "OutputCX", fallback=0) or cp.getint("Video", "BaseCX", fallback=0)
        h = cp.getint("Video", "OutputCY", fallback=0) or cp.getint("Video", "BaseCY", fallback=0)
        kind = cp.getint("Video", "FPSType", fallback=0)
        if kind == 1:
            fps = float(cp.getint("Video", "FPSInt", fallback=30))
        elif kind == 2:
            fps = cp.getint("Video", "FPSNum", fallback=30) / max(1, cp.getint("Video", "FPSDen", fallback=1))
        else:
            fps = float(cp.get("Video", "FPSCommon", fallback="30") or 30)
    except (ValueError, configparser.Error):
        return None
    if not (w and h):
        w, h = 1920, 1080  # OBS's own default when it was never changed
    return w, h, fps


def _scene_file(root: Path) -> Path | None:
    """The scene collection OBS opens with."""
    scenes = root / "basic" / "scenes"
    for name in ("user.ini", "global.ini"):
        f = _ini(root / name).get("Basic", "SceneCollectionFile", fallback="")
        if f and (scenes / f"{f}.json").is_file():
            return scenes / f"{f}.json"
    found = sorted(scenes.glob("*.json")) if scenes.is_dir() else []
    return found[0] if found else None


def _add_capture_if_empty(scene_file: Path) -> bool:
    """A scene collection with no picture source: add Display capture and Game capture (any full-screen game,
    on top) to its current scene. True when something was added."""
    data = json.loads(scene_file.read_text(encoding="utf-8-sig"))
    sources = data.get("sources", [])
    if any(src.get("id") != "scene" and src.get("id") != "group" for src in sources):
        return False  # the streamer has sources: hands off
    current = data.get("current_scene") or (sources[0]["name"] if sources else "Scene")
    scene = next((src for src in sources if src.get("id") == "scene" and src.get("name") == current), None)
    if scene is None:
        scene = {"id": "scene", "versioned_id": "scene", "name": current, "uuid": str(uuid.uuid4()), "settings": {"items": []}}
        sources.append(scene)
        data.setdefault("scene_order", []).append({"name": current})
        data["current_scene"] = data["current_program_scene"] = current
    items = scene.setdefault("settings", {}).setdefault("items", [])
    for n, (kind, name, settings) in enumerate(
        (("monitor_capture", "Display capture", {}), ("game_capture", "Game capture", {"capture_mode": "any_fullscreen"})), 1
    ):
        sid = str(uuid.uuid4())
        sources.append({"id": kind, "versioned_id": kind, "name": name, "uuid": sid, "settings": settings})
        items.append({"name": name, "source_uuid": sid, "visible": True, "id": n})
    scene["settings"]["id_counter"] = len(items)
    data["sources"] = sources
    _backup(scene_file)
    scene_file.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return True


def apply(server: str, env: Mapping[str, str] | None = None, running: Callable[[], bool] = obs_running) -> str:
    """Points OBS at StreamSafe. Returns "" or why not: running / no_obs / error."""
    root = config_dir(env)
    prof = profile_dir(root) if root else None
    if root is None or prof is None:
        return "no_obs"
    if running():
        return "running"
    try:
        svc = prof / "service.json"
        old: dict = {}
        with contextlib.suppress(OSError, ValueError):
            old = json.loads(svc.read_text(encoding="utf-8-sig"))
        settings = dict(old.get("settings", {})) if old.get("type") == "rtmp_custom" else {}
        settings.update({"server": server, "key": "", "use_auth": False, "bwtest": False})
        _write_json(svc, {"type": "rtmp_custom", "settings": settings})
        scene_file = _scene_file(root)
        if scene_file is not None:
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError, AttributeError):
                _add_capture_if_empty(scene_file)
    except (OSError, ValueError):
        return "error"
    return ""


def restore(env: Mapping[str, str] | None = None, running: Callable[[], bool] = obs_running) -> str:
    """Puts back the streamer's own OBS stream settings as they were before StreamSafe. "" or running / no_backup."""
    root = config_dir(env)
    prof = profile_dir(root) if root else None
    keep = prof / ("service.json" + BACKUP) if prof else None
    if keep is None or not keep.is_file():
        return "no_backup"
    if running():
        return "running"
    try:
        shutil.copy2(keep, prof / "service.json")  # type: ignore[operator]
        keep.unlink()  # the next "set up" keeps a fresh copy of whatever the streamer has then
    except OSError:
        return "error"
    return ""


def launch(exe: Path, start_streaming: bool = True, popen: Callable[..., object] = subprocess.Popen) -> bool:
    """Opens OBS (from its own folder, as OBS needs) and lets it start streaming to StreamSafe by itself."""
    args = [str(exe)] + (["--startstreaming"] if start_streaming else [])
    try:
        popen(args, cwd=str(exe.parent), close_fds=True)
    except OSError:
        return False
    return True
