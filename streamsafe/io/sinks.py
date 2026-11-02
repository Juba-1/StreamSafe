"""Where protected frames go.

  * ``VideoFileSink``  - an MP4 file (optionally downscaled; review copies)
  * ``FFmpegSink``     - H.264 to anything FFmpeg can write: ``rtmp://`` / ``rtmps://`` (Twitch,
                         Kick, YouTube ingest), ``srt://`` (back into OBS as a Media Source), a file
  * ``WindowSink``     - an on-screen window (OBS: Window Capture of "StreamSafe Output")
``ThreadedSink`` runs any sink on its own thread so a slow encoder never delays the renderer.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from typing import Protocol

import cv2
import numpy as np

WINDOW_TITLE = "StreamSafe Output"


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
    """Raw BGR frames -> FFmpeg -> H.264 at ``url``. The container follows the URL scheme."""

    def __init__(
        self,
        url: str,
        size: tuple[int, int],
        fps: float = 30.0,
        encoder: str = "libx264",
        bitrate: str = "6000k",
        ffmpeg: str = "ffmpeg",
    ) -> None:
        exe = shutil.which(ffmpeg)
        if exe is None:
            raise FileNotFoundError("ffmpeg not found on PATH")
        w, h = size
        fmt = "flv" if url.startswith(("rtmp://", "rtmps://")) else "mpegts" if url.startswith(("srt://", "udp://")) else None
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
               "-c:v", encoder, "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
               "-g", str(int(2 * fps)), "-pix_fmt", "yuv420p"]  # fmt: skip
        if encoder == "libx264":
            cmd += ["-preset", "veryfast", "-tune", "zerolatency"]
        if fmt:
            cmd += ["-f", fmt]
        self._proc = subprocess.Popen([*cmd, url], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise ConnectionError("ffmpeg output stopped") from exc

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class WindowSink:
    """Needs an OpenCV build with GUI support (``opencv-python``, not ``-headless``)."""

    def __init__(self, title: str = WINDOW_TITLE) -> None:
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def write(self, frame: np.ndarray) -> None:
        cv2.imshow(self.title, frame)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(self.title)


class MultiSink:
    def __init__(self, sinks: list[FrameSink]) -> None:
        self.sinks = sinks

    def write(self, frame: np.ndarray) -> None:
        for sink in self.sinks:
            sink.write(frame)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


class ThreadedSink:
    """Writes on a background thread. Frames are passed by reference: callers must not modify a
    frame after writing it (the engine never does - each output frame is a fresh buffer)."""

    def __init__(self, sink: FrameSink, max_queue: int = 64) -> None:
        self.sink = sink
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max_queue)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="sink", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while (frame := self._queue.get()) is not None:
            if self._error is None:
                try:
                    self.sink.write(frame)
                except BaseException as exc:  # reported to the caller on the next write
                    self._error = exc

    def write(self, frame: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put(frame)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self.sink.close()
        if self._error is not None:
            raise self._error


def open_sink(spec: str, size: tuple[int, int], fps: float = 30.0, preview_size: tuple[int, int] = (960, 540)) -> FrameSink:
    """``window`` -> on-screen window; ``scheme://...`` -> FFmpeg stream; else an MP4 review file."""
    if spec == "window":
        return WindowSink()
    if "://" in spec:
        return ThreadedSink(FFmpegSink(spec, size, fps))
    return ThreadedSink(VideoFileSink(spec, fps, preview_size))
