"""The window's settings, as one small JSON file in the user's data folder. Never the stream key
(``keystore.py``): only what is harmless to read."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

INPUT_CAMERA, INPUT_SRT, INPUT_FILE = "camera", "srt", "file"
OUTPUT_WINDOW, OUTPUT_RTMP, OUTPUT_RECORD = "window", "rtmp", "record"
DEFAULT_CAMERA = "OBS Virtual Camera"


@dataclass
class Settings:
    lang: str = "ar"  # "ar" / "en"
    # INPUT_CAMERA: a capture device (OBS Virtual Camera, a capture card); INPUT_SRT: OBS streams to us over SRT;
    # INPUT_FILE: a video that is already recorded (with OUTPUT_RECORD: "protect a finished video")
    input: str = INPUT_CAMERA
    camera: str = DEFAULT_CAMERA
    camera_sound: str = ""  # the sound input that belongs to the device (a capture card); "" = picture only
    input_file: str = ""
    srt_port: int = 9000
    # OUTPUT_WINDOW: a window OBS captures; OUTPUT_RTMP: we stream to the platform; OUTPUT_RECORD: a video file
    output: str = OUTPUT_WINDOW
    platform: str = "custom"  # ingest.PLATFORMS: which preset fills the address
    rtmp_url: str = ""  # the ingest address WITHOUT the stream key ("" = the platform's preset)
    record_dir: str = ""  # where recordings go ("" = the user's Videos folder)
    remember_key: bool = False
    delay: float = 2.0
    update_serial: int = 0  # the highest update-manifest serial seen (replay protection)
    translate_ar: bool = False  # live English -> Arabic translation of on-screen text (experimental, off by default)
    nvidia_declined: bool = False  # the streamer said no to the NVIDIA download: do not ask at every start
    brand_image: str = ""  # the streamer's / sponsor's logo shown on the full-screen cover (a copy in the data folder)

    def problems(self) -> list[str]:
        """Message ids (``i18n``) of what must be fixed before a stream can start. The key is checked by the caller."""
        out = []
        if self.input == INPUT_SRT and not 1024 <= self.srt_port <= 65535:
            out.append("bad_port")
        if self.input == INPUT_CAMERA and not self.camera.strip():
            out.append("bad_camera")
        if self.input == INPUT_FILE and not os.path.isfile(self.input_file):
            out.append("bad_input_file")
        if self.input == INPUT_FILE and self.output != OUTPUT_RECORD:
            out.append("bad_file_output")
        if self.output == OUTPUT_RTMP:
            from streamsafe.app import ingest

            out += [p for p in ingest.resolve(self.rtmp_url, "k" * 8, self.platform).problems if p != "bad_key"]
        if not 0.5 <= self.delay <= 10:
            out.append("bad_delay")
        return out


def valid_rtmp_url(url: str) -> bool:
    url = url.strip()
    rest = url.split("://", 1)[-1]
    return url.startswith(("rtmp://", "rtmps://")) and bool(rest.strip("/")) and not any(c.isspace() or c in '"<>' for c in url)


def valid_key(key: str) -> bool:
    return bool(key) and len(key) <= 512 and not any(c.isspace() or c in '"<>/\\' for c in key)


def load(path: Path) -> Settings:
    """The stored settings; anything missing, unknown or of the wrong type falls back to its default."""
    s = Settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return s
    if isinstance(raw, dict):
        for f in fields(s):
            value = raw.get(f.name)
            default = getattr(s, f.name)
            if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if type(value) is type(default):
                setattr(s, f.name, value)
    if s.lang not in ("ar", "en"):
        s.lang = "ar"
    if s.input not in (INPUT_CAMERA, INPUT_SRT, INPUT_FILE):
        s.input = INPUT_CAMERA
    if s.output not in (OUTPUT_WINDOW, OUTPUT_RTMP, OUTPUT_RECORD):
        s.output = OUTPUT_WINDOW
    return s


def save(path: Path, settings: Settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
