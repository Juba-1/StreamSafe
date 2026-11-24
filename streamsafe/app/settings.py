"""The window's settings, as one small JSON file in the user's data folder. Never the stream key
(``keystore.py``): only what is harmless to read."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

INPUT_CAMERA, INPUT_SRT, INPUT_FILE = "camera", "srt", "file"
RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440), "2160p": (3840, 2160)}
FPS_CHOICES = (30, 60)
OUTPUT_WINDOW, OUTPUT_RTMP, OUTPUT_RECORD = "window", "rtmp", "record"
COVER_PEOPLE, COVER_SCREEN = "people", "screen"
# the sensitivity slider (a threshold scale; higher = fewer covers). Measured 24 Nov on 183 held-out GTA 6 crops
# (Ali's labels): 1.4 caught only 1-2 of 28 borderline revealing women, so 1.0 is the default and 1.5 the light end
SENS_MIN, SENS_MAX, SENS_DEFAULT = 0.7, 1.5, 1.0
OLD_DEFAULT = 1.4  # the default until 3.0.0.37: such saved settings move to the new default
DEFAULT_CAMERA = "OBS Virtual Camera"


@dataclass
class Settings:
    lang: str = "ar"  # "ar" / "en"
    # INPUT_CAMERA: a capture device (OBS Virtual Camera, a capture card); INPUT_SRT: OBS streams to us over SRT;
    # INPUT_FILE: a video that is already recorded (with OUTPUT_RECORD: "protect a finished video").
    # The defaults are the Home tab's three steps: OBS sends to us (SRT), we send to the platform.
    input: str = INPUT_SRT
    camera: str = DEFAULT_CAMERA
    camera_sound: str = ""  # the sound input that belongs to the device (a capture card); "" = picture only
    input_file: str = ""
    srt_port: int = 9000
    # OUTPUT_WINDOW: a window OBS captures; OUTPUT_RTMP: we stream to the platform; OUTPUT_RECORD: a video file
    output: str = OUTPUT_RTMP
    platform: str = "custom"  # ingest.PLATFORMS: which preset fills the address
    rtmp_url: str = ""  # the ingest address WITHOUT the stream key ("" = the platform's preset)
    platform2: str = ""  # an extra platform streamed at the same time through the protection ("" = none)
    rtmp_url2: str = ""
    record_dir: str = ""  # where recordings go ("" = the user's Videos folder)
    remember_key: bool = False
    delay: float = 0.0  # seconds; 0 = automatic (engine.delay_for: 2 s on the fast NVIDIA mode, else 3 s)
    resolution: str = "1080p"  # of the live picture (RESOLUTIONS); a recorded video keeps its own
    fps: int = 60  # of the live picture: 30 or 60 (a recorded video keeps its own)
    obs_kbps: int = 0  # the streamer's own OBS bitrate: we never send more than it (their internet carries it)
    obs_size: str = ""  # OBS's own output size ("1080x1920" for a vertical TikTok canvas), set at every start
    follow_obs: bool = True  # live from OBS: resolution and fps follow OBS's own video settings at every start
    update_serial: int = 0  # the highest update-manifest serial seen (replay protection)
    translate_ar: bool = False  # live English -> Arabic translation of on-screen text (experimental, off by default)
    nvidia_declined: bool = False  # the streamer said no to the NVIDIA download: do not ask at every start
    sens_scale: float = SENS_DEFAULT  # the live sensitivity slider; > 1 only clearer cases, < 1 anything possible
    zones: list = field(default_factory=list)  # [[x0, y0, x1, y1], ...] fractions: the webcam, never dressed
    dress_everyone: bool = False  # ads: every character, men too, wears the brand garment (switch, live)
    cover_mode: str = COVER_PEOPLE  # COVER_PEOPLE: cover the people who need it; COVER_SCREEN: cover the whole screen
    panic_button: str = "F9"  # the key that covers the whole screen at once (pressed again: back); "" = off
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
            if self.platform2:
                out += [p + "2" for p in ingest.resolve(self.rtmp_url2, "k" * 8, self.platform2).problems if p != "bad_key"]
        if self.delay != 0 and not 0.5 <= self.delay <= 10:
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
        s.input = INPUT_SRT
    if s.output not in (OUTPUT_WINDOW, OUTPUT_RTMP, OUTPUT_RECORD):
        s.output = OUTPUT_RTMP
    try:
        s.sens_scale = SENS_DEFAULT if float(s.sens_scale) == OLD_DEFAULT else min(SENS_MAX, max(SENS_MIN, float(s.sens_scale)))
    except (TypeError, ValueError):
        s.sens_scale = SENS_DEFAULT
    from streamsafe.runtime.control import parse

    s.zones = [list(z) for z in parse({"zones": s.zones if isinstance(s.zones, list) else []}).zones]
    if s.cover_mode not in (COVER_PEOPLE, COVER_SCREEN):
        s.cover_mode = COVER_PEOPLE
    if s.resolution not in RESOLUTIONS:
        s.resolution = "1080p"
    if s.fps not in FPS_CHOICES:
        s.fps = 60
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
