"""Checks before a stream that warn, never block: things on the streamer's PC that can spoil a protected stream
without StreamSafe seeing them. Each returns i18n keys; any failure to read is silence, not an alarm.

* ``multistream``: an OBS plugin that sends to platforms by itself (Aitum Multistream, Multiple RTMP outputs, ...)
  bypasses StreamSafe: whatever it sends is NOT protected.
* ``obs_delay``: OBS's own stream delay adds to ours (works, the viewers are just further behind).
* ``blocked_by_windows``: Windows 11's Smart App Control refused one of our files (shown with the way out).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from pathlib import Path

from streamsafe.app import obs_setup

# folder / file names of OBS plugins that stream to platforms on their own
MULTI_NAMES = ("aitum-multistream", "aitum-multi", "obs-multi-rtmp", "multi-rtmp", "multistream", "vertical-canvas")


def multistream(env: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    places: list[Path] = []
    exe = obs_setup.find_obs(env)
    if exe is not None:
        places.append(exe.parents[2] / "obs-plugins" / "64bit")  # <obs>/bin/64bit/obs64.exe -> <obs>/obs-plugins/64bit
    for var, tail in (("ProgramData", "obs-studio/plugins"), ("APPDATA", "obs-studio/plugins")):
        if env.get(var):
            places.append(Path(env[var]) / tail)
    root = obs_setup.config_dir(env)
    if root is not None:
        places.append(root / "plugin_config")
    for folder in places:
        with contextlib.suppress(OSError):
            if folder.is_dir() and any(any(n in p.name.lower() for n in MULTI_NAMES) for p in folder.iterdir()):
                return ["pre_multistream"]
    return []


def obs_delay(env: Mapping[str, str] | None = None) -> list[str]:
    root = obs_setup.config_dir(env)
    prof = obs_setup.profile_dir(root) if root else None
    if prof is None:
        return []
    cp = obs_setup._ini(prof / "basic.ini")
    return ["pre_obs_delay"] if cp.get("Output", "DelayEnable", fallback="false").lower() == "true" else []


def blocked_by_windows(exc: BaseException) -> bool:
    """The error Windows gives when an Application Control policy (Smart App Control) refuses a file."""
    return getattr(exc, "winerror", None) == 4551 or "4551" in str(exc)


def warnings(env: Mapping[str, str] | None = None) -> list[str]:
    out: list[str] = []
    for check in (lambda: multistream(env), lambda: obs_delay(env)):
        with contextlib.suppress(Exception):
            out += check()
    return out
