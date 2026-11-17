"""Where frames come from. Every source yields BGR uint8 frames of one fixed 16:9 size.

  * ``VideoFileSource``  - a video file, resampled to the engine's frame rate (tests, benchmarks)
  * ``FFmpegSource``     - anything FFmpeg can read live: ``srt://`` (OBS "Custom Output" / SRT
                           stream), ``rtmp://``, ``udp://``, HTTP / HLS ...
  * ``DeviceSource``     - a capture device by index; by name on Windows ("OBS Virtual Camera": OBS Studio
                           -> Virtual Camera, output type "Source" = just the game capture) through FFmpeg
``live`` tells the runner whether the source paces itself (live) or must be paced (file).

Sound: ``FFmpegSource`` also carries the input's sound (game + mic as OBS mixed them). It is addressed by frame:
``frame_index`` is the number of the frame ``frames()`` just gave and ``audio_chunk(frame_index)`` its sound, so
picture and sound stay one unit through the delay, drops included (how: ``avpipe.py``).
"""

from __future__ import annotations

import collections
import contextlib
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol

import cv2
import numpy as np

from streamsafe.io import link
from streamsafe.io.avpipe import AUDIO_RATE, AudioBuffer, AviReader, row_bytes

DEFAULT_SIZE = (1920, 1080)
LIVE_SCHEMES = {"srt", "udp", "rtp", "rtmp", "rtmps", "rtsp", "tcp", "http", "https"}


class FrameSource(Protocol):
    size: tuple[int, int]  # (width, height) of every yielded frame
    live: bool

    def frames(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


def fit_16x9(width: int, height: int) -> tuple[int, int]:
    """The engine draws garments at a fixed 16:9 analysis grid; other aspect ratios are resized."""
    return (width, height) if width * 9 == height * 16 and width > 0 else DEFAULT_SIZE


class VideoFileSource:
    live = False

    def __init__(self, path: str, fps: float = 30.0, max_seconds: float = 1e9, size: tuple[int, int] | None = None) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self.fps = fps
        self.max_seconds = max_seconds
        native = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.size = fit_16x9(*(size or native))
        self._step = (self._cap.get(cv2.CAP_PROP_FPS) or fps) / fps

    def frames(self) -> Iterator[np.ndarray]:
        """Frames resampled to ``fps`` (duplicates or skips source frames as needed)."""
        pos, read, n, frame = 0.0, 0, 0, None
        while n / self.fps < self.max_seconds:
            while read <= int(pos):
                ok, frame = self._cap.read()
                read += 1
                if not ok:
                    return
            yield _fit(frame, self.size)
            n += 1
            pos += self._step

    def close(self) -> None:
        self._cap.release()


class _LatestFrames:
    """Reads a live feed on its own thread into a small buffer that drops the OLDEST frames when
    the consumer is not keeping up (e.g. while the models load): a live feed must never back up
    into the sender (OBS) and the engine always continues from the newest picture."""

    def __init__(self, read_one: Callable[[], Any], maxsize: int = 8) -> None:
        self._read_one = read_one  # a frame (or anything that carries one); None = the feed ended
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="source", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            frame = self._read_one()
            while True:
                try:
                    self._queue.put_nowait(frame)
                    break
                except queue.Full:
                    with contextlib.suppress(queue.Empty):
                        self._queue.get_nowait()
                        self.dropped += 1
            if frame is None:
                return

    def __iter__(self) -> Iterator[Any]:
        while (frame := self._queue.get()) is not None:
            yield frame


_VIDEO_STREAM = re.compile(r"Stream #\d+:\d+.*?: Video: .*?(\d{3,5})x(\d{3,5})")
_FPS = re.compile(r"([\d.]+) (?:fps|tbr)")
_KBPS = re.compile(r"(\d+) kb/s")
# FFmpeg's informational lines (stream description, metadata): not reasons of a failure
_INFO_LINE = re.compile(
    r"^\s*(Input #|Output #|Stream #|Metadata:|Duration:|Program |Stream mapping|Press \[q\]|\w+\s*:\s"
    r"|\[\w+ @ [0-9a-fx]+\] (?!.*(?:rror|ail|efused|in use)))"
)


def sender_format(line: str) -> dict[str, float] | None:
    """What a sender (OBS) sends, from FFmpeg's description of the incoming video stream; None for other lines."""
    m = _VIDEO_STREAM.search(line)
    if not m:
        return None
    out: dict[str, float] = {"width": int(m.group(1)), "height": int(m.group(2))}
    rate = _FPS.search(line)
    if rate:
        out["fps"] = float(rate.group(1))
    kbps = _KBPS.search(line)
    if kbps:
        out["kbps"] = int(kbps.group(1))
    return out


class FFmpegSource:
    """Decodes any FFmpeg input to raw BGR frames (and its sound to PCM) through a pipe.

    OBS example: Settings -> Stream -> Service "Custom", Server ``srt://127.0.0.1:9000`` and run
    StreamSafe with ``--input "srt://127.0.0.1:9000?mode=listener"``.
    Network inputs are live (newest frame wins); file paths / ``file:`` URLs are read in order.

    ``audio``: also read the input's first audio track (an input without one works the same, ``has_audio``
    stays False). Picture and sound are put on the sender's timestamps from one common zero - the picture by
    ``fps=...:start_time=0``, the sound by ``aresample=async:first_pts=0`` (fills gaps, trims overlaps, follows a
    drifting sender clock) - so the sound of frame ``k`` is always the samples ``k * rate / fps`` onwards: tied
    to the picture's own clock, not to arrival times. False: picture only, raw frames on the pipe as before.

    A listener (``srt://...?mode=listener``) stays a listener: FFmpeg's own listener takes one caller and ends
    when that caller goes, so it is started again at once (``relisten``) - OBS may connect late, stop, and
    connect again without anything being restarted by hand. Frames keep one running number over all
    connections, and the sound of a frame from an earlier connection is never looked up in a later one (it is
    silence): sound can not land on the wrong picture. What the link does is said in ``STATUS`` lines
    (``io/link.py``). ``error`` holds why the input gave up ("" when it simply ended).
    """

    QUICK_S = 3.0  # an FFmpeg that ends this soon without one frame did not open its input at all
    GIVE_UP_AFTER = 3  # ... this many times in a row: the input fails (port busy), nothing is retried for ever

    def __init__(
        self,
        url: str,
        size: tuple[int, int] = DEFAULT_SIZE,
        fps: float = 30.0,
        ffmpeg: str = "ffmpeg",
        input_args: tuple[str, ...] = (),
        live: bool | None = None,
        audio: bool = True,
        relisten: bool | None = None,
        spawn: Callable[..., Any] | None = None,
    ) -> None:
        exe = shutil.which(ffmpeg)
        if exe is None and spawn is None:
            raise FileNotFoundError("ffmpeg not found on PATH")
        self.size, self.fps, self.url = fit_16x9(*size), fps, url
        self.live = url.split("://", 1)[0].lower() in LIVE_SCHEMES if live is None else live
        self.frame_index = -1  # number (over all connections) of the frame frames() gave last
        self.dropped_audio_frames = 0
        self.error = ""
        self.connections = 0  # how many times a sender started sending
        self._port = link.listener_port(url)
        self.relisten = self._port is not None if relisten is None else relisten
        w, h = self.size
        # no "-fflags nobuffer": with FFmpeg 6.1 it decodes nothing from mp4 files or SRT/mpegts (tested).
        # "fps=" makes any input rate (OBS virtual camera at 60 fps) the engine's rate
        # a listener logs at "info": FFmpeg then says what OBS sends (size, frame rate, bitrate), passed on as a
        # STATUS line so the window can warn when the protection runs smaller or slower than OBS sends
        level = ["-loglevel", "info", "-nostats"] if self._port is not None else ["-loglevel", "error"]
        cmd = [exe or ffmpeg, "-hide_banner", *level, *input_args, "-i", url]
        if audio:
            # a small interleave window: if the sound stalls, the picture is not held back for it
            cmd += ["-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", f"scale={w}:{h},fps={fps}:start_time=0", "-pix_fmt", "bgr24", "-c:v", "rawvideo",
                    "-af", f"aresample={AUDIO_RATE}:async=1000:first_pts=0,aformat=sample_fmts=s16:channel_layouts=stereo",
                    "-c:a", "pcm_s16le", "-max_interleave_delta", "100000", "-flush_packets", "1", "-f", "avi", "pipe:1"]  # fmt: skip
        else:
            cmd += ["-an", "-vf", f"scale={w}:{h},fps={fps}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        self._cmd, self._audio_on = cmd, audio
        self._spawn = spawn or subprocess.Popen
        self._closed = threading.Event()
        self._keep_s = 30.0
        self._n_read = 0
        self._fails = 0
        self._errors: collections.deque[str] = collections.deque(maxlen=6)
        self._audio: tuple[int, AudioBuffer | None] = (0, None)  # (number of this connection's first frame, its sound)
        self._start()
        self._live_reader = _LatestFrames(self._read_one) if self.live else None  # start draining at once

    def _start(self) -> None:
        w, h = self.size
        self._errors.clear()
        self._proc = self._spawn(self._cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=w * h * 3)
        assert self._proc.stdout is not None
        self._avi = AviReader(self._proc.stdout) if self._audio_on else None
        sound = AudioBuffer(self.fps, self._keep_s) if self._audio_on else None
        self._audio = (self._n_read, sound)
        self._sound = sound
        self._got = False
        self._began = time.monotonic()
        threading.Thread(target=self._drain_errors, args=(self._proc,), name="source-errors", daemon=True).start()
        if self._port is not None:
            link.status("waiting_obs", port=self._port)

    def _drain_errors(self, proc: Any) -> None:
        """FFmpeg's own messages: passed on (the engine log) and the last ones kept as the reason of a failure."""
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return
        with contextlib.suppress(OSError, ValueError):
            for raw in stream:
                line = raw.decode("utf-8", "replace").rstrip() if isinstance(raw, bytes) else str(raw).rstrip()
                if line:
                    sent = sender_format(line)
                    if sent:
                        link.status("obs_format", **sent)
                    if not _INFO_LINE.match(line):
                        self._errors.append(line)
                    print(f"INPUT {line}", file=sys.stderr, flush=True)

    @property
    def has_audio(self) -> bool:
        """The input has a sound track (known once the first frame has arrived)."""
        return self._avi is not None and self._avi.has_audio

    def audio_keep(self, seconds: float) -> None:
        """How much sound is kept for frames still waiting to be shown (the runner: delay + margin)."""
        self._keep_s = seconds
        if self._sound is not None:
            self._sound.keep_s = seconds

    def audio_chunk(self, index: int) -> np.ndarray | None:
        """Sound of frame ``index`` (int16 [n, 2] at 48 kHz); None when there is none (no track, not arrived, gone,
        or the frame came over an earlier connection than the one sending now)."""
        first, sound = self._audio
        if sound is None or index < first:
            return None
        return sound.chunk(index - first)

    def _read_frame(self) -> np.ndarray | None:
        w, h = self.size
        assert self._proc.stdout is not None
        if self._avi is None:
            buf = self._proc.stdout.read(w * h * 3)
        else:
            assert self._sound is not None
            while (chunk := self._avi.next()) is not None and chunk[0] == "audio":
                self._sound.append(chunk[1])
            buf = chunk[1] if chunk is not None else b""
            if len(buf) == row_bytes(w) * h != w * h * 3:  # rows padded to 4 bytes (never for 16:9 sizes)
                buf = np.frombuffer(buf, np.uint8).reshape(h, -1)[:, : w * 3].tobytes()
        if len(buf) != w * h * 3:
            return None  # this FFmpeg ended
        return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()

    def _read_one(self) -> tuple[int, np.ndarray] | None:
        while True:
            frame = self._read_frame()
            if frame is not None:
                if not self._got:
                    self._got, self._fails = True, 0
                    self.connections += 1
                    if self._port is not None:
                        link.status("obs_connected")
                self._n_read += 1
                return self._n_read - 1, frame
            if self._closed.is_set():
                return None
            self._reap()
            quick = time.monotonic() - self._began < self.QUICK_S
            why = link.reason_of(" ".join(self._errors), "input_ended")
            if self._got:
                if self._port is not None:
                    link.status("obs_disconnected")
            elif quick or not self.relisten:
                self._fails += 1
            if not self.relisten or self._fails >= self.GIVE_UP_AFTER:
                if not self._got and (self.live or self._n_read == 0):  # never opened: say why
                    self.error = why
                    link.status("input_failed", reason=why, detail=" ".join(self._errors)[-300:])
                return None
            if self._closed.wait(0.3 if self._got else 0.5):
                return None
            self._start()

    def _reap(self) -> None:
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=3)

    def frames(self) -> Iterator[np.ndarray]:
        numbered = self._live_reader if self._live_reader is not None else iter(self._read_one, None)
        for self.frame_index, frame in numbered:
            yield frame

    def close(self) -> None:
        self._closed.set()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def parse_dshow_devices(text: str) -> dict[str, list[str]]:
    """``ffmpeg -list_devices true -f dshow -i dummy`` -> {"video": [names], "audio": [names]} (both FFmpeg
    formats: the newer `"name" (video)` lines and the older "DirectShow video devices" sections)."""
    out: dict[str, list[str]] = {"video": [], "audio": []}
    section = ""
    for line in text.splitlines():
        low = line.lower()
        if "directshow video devices" in low:
            section = "video"
        elif "directshow audio devices" in low:
            section = "audio"
        m = re.search(r'"([^"]+)"(?:\s+\((video|audio|none)\))?\s*$', line)
        if not m or "alternative name" in low or m.group(1).startswith("@device"):
            continue
        kind = m.group(2) if m.group(2) in ("video", "audio") else section
        if kind and m.group(1) not in out[kind]:
            out[kind].append(m.group(1))
    return out


def list_devices(ffmpeg: str = "ffmpeg") -> dict[str, list[str]]:
    """The capture devices Windows offers (cameras, capture cards, the OBS Virtual Camera; microphones and the
    sound inputs of capture cards). Empty lists anywhere else or when FFmpeg cannot be asked."""
    exe = shutil.which(ffmpeg)
    if exe is None or sys.platform != "win32":
        return {"video": [], "audio": []}
    try:
        done = subprocess.run(
            [exe, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, timeout=20, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return {"video": [], "audio": []}
    return parse_dshow_devices(done.stderr.decode("utf-8", "replace"))


def windows_device(name: str, size: tuple[int, int] = DEFAULT_SIZE, fps: float = 30.0, sound: str = "", audio: bool = True) -> FFmpegSource:
    """A DirectShow device by its name on Windows ("OBS Virtual Camera", a capture card), through FFmpeg: OpenCV's
    DirectShow backend opens devices by index only (Fable review), and an index changes when a webcam is plugged in.
    ``sound``: the name of the device's sound input (a capture card carries the console's sound as a second
    DirectShow device); "" = picture only (a camera)."""
    args = ("-f", "dshow", "-rtbufsize", "256M")
    if sound and audio:
        return FFmpegSource(f"video={name}:audio={sound}", size, fps, input_args=args, live=True, audio=True)
    return FFmpegSource(f"video={name}", size, fps, input_args=args, live=True, audio=False)


class DeviceSource:
    """A capture device by index ("0"). Frames arriving faster than ``fps`` (a 60 fps device) are thinned to
    ``fps`` so the engine's frame count and the clock agree."""

    live = True

    def __init__(self, device: str, size: tuple[int, int] = DEFAULT_SIZE, fps: float = 30.0) -> None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(int(device), backend)
        if not self._cap.isOpened():
            raise ConnectionError(f"cannot open capture device: {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.size = fit_16x9(*size)
        self.fps = fps
        self._next = 0.0
        self._live_reader = _LatestFrames(self._read_one)

    def _read_one(self) -> np.ndarray | None:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return None
            now = time.monotonic()
            if now >= self._next:  # keep this frame; the next is due one period later (no drift)
                self._next = max(self._next + 1.0 / self.fps, now - 0.5 / self.fps)
                return _fit(frame, self.size)

    def frames(self) -> Iterator[np.ndarray]:
        yield from self._live_reader

    def close(self) -> None:
        self._cap.release()


def open_source(
    spec: str, fps: float = 30.0, max_seconds: float = 1e9, size: tuple[int, int] | None = None, audio: bool = True
) -> FrameSource:
    """``device:<index|name>`` -> device (``device:<name>|<sound device>``: with the sound of a capture card);
    ``scheme://...`` -> FFmpeg; ``file:<path>`` -> a video file with its sound; anything else -> video file (picture).

    ``size``: output frame size; None = the file's own size (live inputs default to 1080p).
    ``audio``: FFmpeg inputs also carry their sound (devices and plain video files are picture only)."""
    if spec.startswith("device:"):
        device = spec[len("device:") :]
        if device.isdigit():
            return DeviceSource(device, size or DEFAULT_SIZE, fps)
        if sys.platform != "win32":
            raise ValueError(f"capture devices by name need Windows; use an index here: {spec!r}")
        name, _, sound = device.partition("|")
        return windows_device(name.strip(), size or DEFAULT_SIZE, fps, sound.strip(), audio)
    if spec.startswith("file:") and "://" not in spec:  # a recorded video WITH its sound, read in order
        path = spec[len("file:") :]
        if size is None:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video: {path}")
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
        args = ("-t", f"{max_seconds:g}") if max_seconds < 1e8 else ()
        return FFmpegSource(path, size, fps, input_args=args, live=False, audio=audio)
    if "://" in spec:
        return FFmpegSource(spec, size or DEFAULT_SIZE, fps, audio=audio)
    return VideoFileSource(spec, fps, max_seconds, size)


def _fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return frame if frame.shape[1::-1] == size else cv2.resize(frame, size)
