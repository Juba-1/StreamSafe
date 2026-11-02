"""Where frames come from. Every source yields BGR uint8 frames of one fixed 16:9 size.

  * ``VideoFileSource``  - a video file, resampled to the engine's frame rate (tests, benchmarks)
  * ``FFmpegSource``     - anything FFmpeg can read live: ``srt://`` (OBS "Custom Output" / SRT
                           stream), ``rtmp://``, ``udp://``, HTTP / HLS ...
  * ``DeviceSource``     - a capture device, e.g. "OBS Virtual Camera" (OBS Studio -> Virtual
                           Camera, output type "Source" = just the game capture)
``live`` tells the runner whether the source paces itself (live) or must be paced (file).
"""

from __future__ import annotations

import contextlib
import queue
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Protocol

import cv2
import numpy as np

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

    def __init__(self, read_one: Callable[[], np.ndarray | None], maxsize: int = 8) -> None:
        self._read_one = read_one
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=maxsize)
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

    def __iter__(self) -> Iterator[np.ndarray]:
        while (frame := self._queue.get()) is not None:
            yield frame


class FFmpegSource:
    """Decodes any FFmpeg input to raw BGR frames through a pipe.

    OBS example: Settings -> Stream -> Service "Custom", Server ``srt://127.0.0.1:9000`` and run
    StreamSafe with ``--input "srt://127.0.0.1:9000?mode=listener"``.
    Network inputs are live (newest frame wins); file paths / ``file:`` URLs are read in order.
    """

    def __init__(self, url: str, size: tuple[int, int] = DEFAULT_SIZE, fps: float = 30.0, ffmpeg: str = "ffmpeg") -> None:
        exe = shutil.which(ffmpeg)
        if exe is None:
            raise FileNotFoundError("ffmpeg not found on PATH")
        self.size, self.fps, self.url = fit_16x9(*size), fps, url
        self.live = url.split("://", 1)[0].lower() in LIVE_SCHEMES
        w, h = self.size
        # no "-fflags nobuffer": with FFmpeg 6.1 it decodes nothing from mp4 files or SRT/mpegts (tested)
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-i", url,
               "-an", "-vf", f"scale={w}:{h},fps={fps}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]  # fmt: skip
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, bufsize=w * h * 3)
        self._live_reader = _LatestFrames(self._read_one) if self.live else None  # start draining at once

    def _read_one(self) -> np.ndarray | None:
        w, h = self.size
        assert self._proc.stdout is not None
        buf = self._proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            return None  # stream ended
        return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()

    def frames(self) -> Iterator[np.ndarray]:
        if self._live_reader is not None:
            yield from self._live_reader
            return
        while (frame := self._read_one()) is not None:
            yield frame

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class DeviceSource:
    """A capture device by index ("0") or by name ("OBS Virtual Camera", Windows only)."""

    live = True

    def __init__(self, device: str, size: tuple[int, int] = DEFAULT_SIZE, fps: float = 30.0) -> None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        target = int(device) if device.isdigit() else f"video={device}"
        self._cap = cv2.VideoCapture(target, backend)
        if not self._cap.isOpened():
            raise ConnectionError(f"cannot open capture device: {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.size = fit_16x9(*size)
        self.fps = fps
        self._live_reader = _LatestFrames(self._read_one)

    def _read_one(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return _fit(frame, self.size) if ok else None

    def frames(self) -> Iterator[np.ndarray]:
        yield from self._live_reader

    def close(self) -> None:
        self._cap.release()


def open_source(spec: str, fps: float = 30.0, max_seconds: float = 1e9, size: tuple[int, int] | None = None) -> FrameSource:
    """``device:<index|name>`` -> device; ``scheme://...`` -> FFmpeg; anything else -> video file.

    ``size``: output frame size; None = the file's own size (live inputs default to 1080p)."""
    if spec.startswith("device:"):
        return DeviceSource(spec[len("device:") :], size or DEFAULT_SIZE, fps)
    if "://" in spec:
        return FFmpegSource(spec, size or DEFAULT_SIZE, fps)
    return VideoFileSource(spec, fps, max_seconds, size)


def _fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return frame if frame.shape[1::-1] == size else cv2.resize(frame, size)
