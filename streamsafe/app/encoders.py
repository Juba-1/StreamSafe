"""Which H.264 encoder the bundled FFmpeg uses for the stream to the platform.

The FFmpeg we ship is an LGPL build (no GPL parts), so it has no libx264. It has the encoders of the graphics
card and of Windows instead: NVIDIA NVENC, AMD AMF, Intel Quick Sync, and Media Foundation (Windows' own, works on
every PC). Each candidate is TRIED on a few test frames, because "listed" does not mean "works on this PC" (an
NVENC encoder is listed on a PC without an NVIDIA card). The first one that works is handed to the engine in the
``ENCODER`` environment variable (read by ``io/sinks.py``). None works: the stream does not start, nothing is
sent.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, MutableMapping, Sequence

CANDIDATES = ("h264_nvenc", "h264_amf", "h264_qsv", "h264_mf", "libx264")
STREAM_SCHEMES = ("rtmp://", "rtmps://", "srt://", "udp://")
TIMEOUT_S = 20

Run = Callable[..., subprocess.CompletedProcess]


def _flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def listed(ffmpeg: str, run: Run = subprocess.run) -> set[str]:
    """The H.264 encoders this FFmpeg lists."""
    try:
        out = run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=TIMEOUT_S, creationflags=_flags()).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {parts[1] for line in out.splitlines() if len(parts := line.split()) >= 2 and parts[1] in CANDIDATES}


def works(ffmpeg: str, encoder: str, run: Run = subprocess.run) -> bool:
    """Does ``encoder`` really encode 10 frames of the size and pixel format the stream uses, on this PC?"""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
        "-frames:v", "10", "-c:v", encoder, "-pix_fmt", "yuv420p", "-f", "null", "-",
    ]  # fmt: skip
    try:
        return run(cmd, capture_output=True, timeout=TIMEOUT_S, creationflags=_flags()).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def choose(ffmpeg: str, run: Run = subprocess.run) -> str:
    """The first working encoder, "" when none works."""
    have = listed(ffmpeg, run)
    return next((e for e in CANDIDATES if e in have and works(ffmpeg, e, run)), "")


def output_of(args: Sequence[str]) -> str:
    for i, arg in enumerate(args):
        if arg == "--output" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--output="):
            return arg.split("=", 1)[1]
    return ""


def choose_into(env: MutableMapping[str, str], args: Sequence[str], run: Run = subprocess.run) -> None:
    """Before ``run``: when the output is a network stream and no ``ENCODER`` is set, find a working one and put
    it into ``env``. Raises SystemExit (with the reason) when there is none."""
    if env.get("ENCODER") or not output_of(args).startswith(STREAM_SCHEMES):
        return
    from streamsafe.app import paths

    ffmpeg = paths.find_ffmpeg(env)
    encoder = choose(ffmpeg, run) if ffmpeg else ""
    if not encoder:
        raise SystemExit("NO_ENCODER no H.264 encoder works on this PC (NVENC, AMF, Quick Sync, Media Foundation). Nothing was streamed.")
    env["ENCODER"] = encoder
