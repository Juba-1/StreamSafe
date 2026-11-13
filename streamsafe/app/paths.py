"""Where the app keeps things, and where it finds what the installer put next to the exe.

Per user, no administrator rights:
  program   %LOCALAPPDATA%\\Programs\\<name>\\           (the installer; read-only for the app)
  data      %LOCALAPPDATA%\\<name>\\                      models, engines, reports, settings, the protected stream key
``STREAMSAFE_DATA_DIR`` moves the data folder (tests, CI, a second disk).
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, MutableMapping
from pathlib import Path

from streamsafe.branding import PRODUCT_NAME

DATA_DIR_ENV = "STREAMSAFE_DATA_DIR"
FFMPEG_EXE = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
# OBS's Lua cannot open a path with non-ASCII letters on Windows (docs/OBS_SETUP.md). A Windows user name may have
# them: the state file then goes to the 8.3 short name of its folder (all ASCII, same per-user folder), and only if
# the volume has no short names to the all-ASCII "Public" folder (every user may write there; no admin rights, no
# ProgramData)
PUBLIC_FALLBACK = r"C:\Users\Public"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """The folder of the installed program (frozen build) or the repository root (running from source)."""
    return Path(sys.executable).resolve().parent if is_frozen() else Path(__file__).resolve().parents[2]


def data_dir(env: MutableMapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get(DATA_DIR_ENV):
        return Path(env[DATA_DIR_ENV])
    base = env.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / PRODUCT_NAME


def models_dir(env: MutableMapping[str, str] | None = None) -> Path:
    return data_dir(env) / "models"


def translate_dir(env: MutableMapping[str, str] | None = None) -> Path:
    """The translation pack (OCR models, translator, fonts): downloaded when the streamer switches Arabic translation on."""
    return data_dir(env) / "translate"


def heads_dir(env: MutableMapping[str, str] | None = None) -> Path:
    """Heads brought by an update (the ones the installer ships are ``bundled_heads_dir``)."""
    return data_dir(env) / "heads"


def bundled_heads_dir() -> Path:
    return app_dir() / "heads"


def reports_dir(env: MutableMapping[str, str] | None = None) -> Path:
    return data_dir(env) / "reports"


def logs_dir(env: MutableMapping[str, str] | None = None) -> Path:
    return data_dir(env) / "logs"


def downloads_dir(env: MutableMapping[str, str] | None = None) -> Path:
    return data_dir(env) / "downloads"


def trt_cache_dir(env: MutableMapping[str, str] | None = None) -> Path:
    return data_dir(env) / "trt_cache"


def nvidia_dir(env: MutableMapping[str, str] | None = None) -> Path:
    return data_dir(env) / "nvidia"


def settings_file(env: MutableMapping[str, str] | None = None) -> Path:
    return data_dir(env) / "settings.json"


def key_file(env: MutableMapping[str, str] | None = None) -> Path:
    return data_dir(env) / "stream_key.bin"


def _short_path(folder: str) -> str:
    import ctypes

    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(folder, buf, len(buf))  # type: ignore[attr-defined]
    return buf.value if 0 < n < len(buf) else ""


def obs_state_file(
    env: MutableMapping[str, str] | None = None, windows: bool | None = None, short_path: Callable[[str], str] | None = None
) -> Path:
    """The heartbeat file the OBS guard script reads: in the user's own data folder, unless that path has
    non-ASCII letters (then its 8.3 short name, or the Public folder; Windows only)."""
    env = os.environ if env is None else env
    path = data_dir(env) / "obs" / "state.txt"
    windows = sys.platform == "win32" if windows is None else windows
    if str(path).isascii() or not windows:
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        short = (short_path or _short_path)(str(path.parent))
    except OSError:
        short = ""
    if short and short.isascii():
        return Path(short) / path.name
    return Path(env.get("PUBLIC") or PUBLIC_FALLBACK) / PRODUCT_NAME / path.name


def obs_script() -> Path:
    return app_dir() / "obs" / "streamsafe_guard.lua"


def bundled_ffmpeg_dir(root: Path | None = None) -> Path | None:
    """The folder of the FFmpeg the installer ships (``<program>/ffmpeg/bin``), or None when it is not there."""
    folder = (root or app_dir()) / "ffmpeg" / "bin"
    return folder if (folder / FFMPEG_EXE).is_file() else None


def use_bundled_ffmpeg(env: MutableMapping[str, str] | None = None, root: Path | None = None) -> Path | None:
    """Put the bundled FFmpeg in front of PATH, so every ``shutil.which("ffmpeg")`` in the engine (inputs, outputs,
    the system check) and every child process finds the bundled one first and a system FFmpeg second."""
    env = os.environ if env is None else env
    folder = bundled_ffmpeg_dir(root)
    if folder is None:
        return None
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    if not parts or Path(parts[0]) != folder:
        env["PATH"] = os.pathsep.join([str(folder), *(p for p in parts if Path(p) != folder)])
    return folder


def find_ffmpeg(env: MutableMapping[str, str] | None = None, root: Path | None = None) -> str | None:
    """The FFmpeg to run: the bundled one first, then PATH. None when there is none."""
    env = os.environ if env is None else env
    folder = bundled_ffmpeg_dir(root)
    if folder is not None:
        return str(folder / FFMPEG_EXE)
    return shutil.which("ffmpeg", path=env.get("PATH"))
