"""Where protected frames go.

  * ``VideoFileSink``  - an MP4 file (optionally downscaled; review copies)
  * ``FFmpegSink``     - H.264 to anything FFmpeg can write: ``rtmp://`` / ``rtmps://`` (Twitch,
                         Kick, YouTube ingest), ``srt://`` (back into OBS as a Media Source), a file
  * ``WindowSink``     - an on-screen window (OBS: Window Capture of "StreamSafe Output")
``ThreadedSink`` runs any sink on its own thread so a slow encoder never delays the renderer.

Sound: a sink with ``takes_audio`` gets each frame's sound with the frame, ``write(frame, audio)``; ``audio`` is
int16 [n, 2] at 48 kHz or None for silence (muted, or no sound known for this frame). The others get the picture.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Protocol

import cv2
import numpy as np

from streamsafe.branding import PRODUCT_NAME
from streamsafe.io.avpipe import AUDIO_RATE, AviWriter

WINDOW_TITLE = f"{PRODUCT_NAME} Output"


class FrameSink(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


class VideoFileSink:
    def __init__(self, path: str, fps: float = 30.0, size: tuple[int, int] = (960, 540)) -> None:
        self.size = size
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._writer.isOpened():
            raise OSError(f"cannot write video: {path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1::-1] != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


class FFmpegSink:
    """Raw BGR frames (+ their sound) -> FFmpeg -> H.264 (+ AAC) at ``url``. The container follows the URL scheme.

    ``audio`` (default): the output always has a sound track - Twitch and YouTube ingest need one - made of each
    frame's own samples, or silence for a frame written without sound. One frame and exactly one frame's worth of
    samples go down one pipe together, so the two cannot drift apart. False: picture only, as before."""

    def __init__(
        self,
        url: str,
        size: tuple[int, int],
        fps: float = 30.0,
        encoder: str = os.environ.get("ENCODER") or "libx264",  # the installed app picks the one that works (app/encoders.py)
        bitrate: str = "6000k",
        ffmpeg: str = "ffmpeg",
        audio: bool = True,
        audio_bitrate: str = "160k",
    ) -> None:
        exe = shutil.which(ffmpeg)
        if exe is None:
            raise FileNotFoundError("ffmpeg not found on PATH")
        w, h = size
        self.takes_audio = audio
        fmt = "flv" if url.startswith(("rtmp://", "rtmps://")) else "mpegts" if url.startswith(("srt://", "udp://")) else None
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y"]
        if audio:  # our own stream, nothing to probe: start at once
            cmd += ["-probesize", "32", "-analyzeduration", "0", "-f", "avi", "-i", "pipe:0"]
        else:
            cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0"]
        cmd += ["-c:v", encoder, "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
                "-g", str(int(2 * fps)), "-pix_fmt", "yuv420p"]  # fmt: skip
        if encoder == "libx264":
            cmd += ["-preset", "veryfast", "-tune", "zerolatency"]
        if audio:
            cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", str(AUDIO_RATE), "-ac", "2"]
        if fmt:
            cmd += ["-f", fmt]
        self._proc = subprocess.Popen([*cmd, url], stdin=subprocess.PIPE)
        assert self._proc.stdin is not None
        self._avi = AviWriter(self._proc.stdin, size, fps) if audio else None

    def write(self, frame: np.ndarray, audio: np.ndarray | None = None) -> None:
        assert self._proc.stdin is not None
        try:
            if self._avi is not None:
                self._avi.write(frame, audio)
            else:
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except OSError as exc:  # BrokenPipeError; on Windows a closed pipe is "Invalid argument"
            raise ConnectionError("ffmpeg output stopped") from exc

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class WindowSink:
    """The window OBS captures (Window Capture "StreamSafe Output"). Needs an OpenCV build with GUI support
    (``opencv-python``, not ``-headless``).

    Fixed size (the output frame, never scaled by dragging). Windows stops drawing a minimised window and OBS
    then captures nothing, so on Windows a minimised window is restored at once without taking the focus
    (``restored`` counts it); the OBS guard also shows the slate whenever the captured window is empty."""

    CHECK_S = 0.25

    def __init__(self, title: str = WINDOW_TITLE) -> None:
        self.title = title
        self.restored = 0
        self._next_check = 0.0
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def write(self, frame: np.ndarray) -> None:
        cv2.imshow(self.title, frame)
        cv2.waitKey(1)
        if sys.platform == "win32" and time.monotonic() >= self._next_check:
            self._next_check = time.monotonic() + self.CHECK_S
            self.restored += _restore_if_minimised(self.title)

    def close(self) -> None:
        cv2.destroyWindow(self.title)


def takes_audio(sink: FrameSink | None) -> bool:
    return bool(getattr(sink, "takes_audio", False))


class MultiSink:
    def __init__(self, sinks: list[FrameSink]) -> None:
        self.sinks = sinks
        self.takes_audio = any(takes_audio(s) for s in sinks)

    def write(self, frame: np.ndarray, audio: np.ndarray | None = None) -> None:
        for sink in self.sinks:
            _write(sink, frame, audio)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


class ThreadedSink:
    """Writes on a background thread. Frames are passed by reference: callers must not modify a
    frame after writing it (the engine never does - each output frame is a fresh buffer)."""

    def __init__(self, sink: FrameSink, max_queue: int = 64) -> None:
        self.sink = sink
        self.takes_audio = takes_audio(sink)
        self._queue: queue.Queue[tuple[np.ndarray, np.ndarray | None] | None] = queue.Queue(maxsize=max_queue)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="sink", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            if self._error is None:
                try:
                    _write(self.sink, *item)
                except BaseException as exc:  # reported to the caller on the next write
                    self._error = exc

    def write(self, frame: np.ndarray, audio: np.ndarray | None = None) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put((frame, audio))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self.sink.close()
        if self._error is not None:
            raise self._error


def _write(sink: FrameSink, frame: np.ndarray, audio: np.ndarray | None) -> None:
    if takes_audio(sink):
        sink.write(frame, audio)  # type: ignore[call-arg]
    else:
        sink.write(frame)


def open_sink(
    spec: str, size: tuple[int, int], fps: float = 30.0, preview_size: tuple[int, int] = (960, 540), audio: bool = True
) -> FrameSink:
    """``window`` -> on-screen window; ``scheme://...`` -> FFmpeg stream (with sound unless ``audio`` is off);
    else an MP4 review file (picture only)."""
    if spec == "window":
        return WindowSink()
    if "://" in spec:
        return ThreadedSink(FFmpegSink(spec, size, fps, audio=audio))
    return ThreadedSink(VideoFileSink(spec, fps, preview_size))


def _restore_if_minimised(title: str) -> int:
    """Windows only: un-minimise the window ``title`` without activating it. 1 if it was minimised."""
    import ctypes

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if hwnd and user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE
        return 1
    return 0
