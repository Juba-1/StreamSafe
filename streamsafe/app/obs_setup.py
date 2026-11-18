"""Set OBS up for StreamSafe by itself, so the streamer does not touch OBS's settings.

What it changes in OBS's current profile (a copy of each file is kept once as ``*.before-streamsafe``):
- Settings > Stream: Custom server = StreamSafe's local address, no key (``service.json``);
- the bitrate OBS sends to StreamSafe: high (it never leaves the PC), keyframe every 2 s
  (``basic.ini`` [SimpleOutput], or ``streamEncoder.json`` in Advanced output mode).

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
from collections.abc import Callable, Mapping
from pathlib import Path

OBS_EXE = "obs64.exe"
OBS_PATHS = (r"C:\Program Files\obs-studio\bin\64bit\obs64.exe", r"C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe")
LOCAL_KBPS = 20000  # OBS -> StreamSafe stays inside the PC: high costs nothing and keeps the picture sharp
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
    for p in (*OBS_PATHS, env.get("STREAMSAFE_OBS_EXE", "")):
        if p and Path(p).is_file():
            return Path(p)
    which = shutil.which(OBS_EXE)
    return Path(which) if which else None


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


def apply(server: str, env: Mapping[str, str] | None = None, running: Callable[[], bool] = obs_running) -> str:
    """Points OBS at StreamSafe. Returns "" or why not: running / no_obs / error."""
    root = config_dir(env)
    prof = profile_dir(root) if root else None
    if prof is None:
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

        ini_path = prof / "basic.ini"
        cp = _ini(ini_path)
        mode = cp.get("Output", "Mode", fallback="Simple")
        if mode == "Advanced":
            enc = prof / "streamEncoder.json"
            data: dict = {}
            with contextlib.suppress(OSError, ValueError):
                data = json.loads(enc.read_text(encoding="utf-8-sig"))
            data.update({"rate_control": "CBR", "bitrate": max(int(data.get("bitrate", 0) or 0), LOCAL_KBPS), "keyint_sec": 2})
            _write_json(enc, data)
        else:
            if not cp.has_section("SimpleOutput"):
                cp.add_section("SimpleOutput")
            now = cp.get("SimpleOutput", "VBitrate", fallback="0")
            cp.set("SimpleOutput", "VBitrate", str(max(int(now) if now.isdigit() else 0, LOCAL_KBPS)))
            _backup(ini_path)
            with ini_path.open("w", encoding="utf-8-sig", newline="\r\n") as f:
                cp.write(f, space_around_delimiters=False)
    except (OSError, ValueError, configparser.Error):
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
