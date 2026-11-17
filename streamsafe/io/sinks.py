"""Where protected frames go.

  * ``VideoFileSink``  - an MP4 file (optionally downscaled; review copies)
  * ``FFmpegSink``     - H.264 to anything FFmpeg can write: ``rtmp://`` / ``rtmps://`` (Twitch,
                         Kick, YouTube ingest), ``srt://`` (back into OBS as a Media Source), a file
  * ``RecordSink``     - the protected picture and sound as a video file ("record only", ``record:<file>``)
  * ``WindowSink``     - an on-screen window (OBS: Window Capture of "StreamSafe Output")
``ThreadedSink`` runs any sink on its own thread so a slow encoder never delays the renderer.

Sound: a sink with ``takes_audio`` gets each frame's sound with the frame, ``write(frame, audio)``; ``audio`` is
int16 [n, 2] at 48 kHz or None for silence (muted, or no sound known for this frame). The others get the picture.
"""

from __future__ import annotations

import collections
import contextlib
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import cv2
import numpy as np

from streamsafe.branding import PRODUCT_NAME
from streamsafe.io import link
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


STREAM_SCHEMES = ("rtmp://", "rtmps://", "srt://", "udp://")
_PROGRESS = re.compile(
    r"^(frame|fps|stream_\d+_\d+_q|bitrate|total_size|out_time_us|out_time_ms|out_time|dup_frames|drop_frames|speed|progress)=(.*)$"
)


class PlatformRefused(ConnectionError):  # noqa: N818 - a refusal, not an error
    """The platform never took the stream (wrong address or key, no network): ``reason`` is a short word
    (``io/link.py``), the text never holds the address or the key."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"the platform did not accept the stream ({reason}) {detail}".strip())
        self.reason = reason


def auto_kbps(size: tuple[int, int], fps: float, network: bool) -> int:
    """A video bitrate that does not make the picture worse than what came in.

    Network (to a platform): what the big platforms take at that size - 1080p30 6000, 1080p60 9000, 1440p 16000,
    4K 30000 kbit/s; the app passes the platform's own figure instead (``app/ingest.py``). A file (recording,
    a protected copy of a video): about 0.2 bits per pixel and frame, at least 20000 kbit/s - visually the same
    as the source for game footage, 1080p60 ~25 Mbit/s, 4K60 ~100 Mbit/s."""
    w, h = size
    if network:
        if h <= 720:
            return 4500 if fps <= 30 else 6000
        if h <= 1080:
            return 6000 if fps <= 30 else 9000
        if h <= 1440:
            return 16000
        return 30000
    return max(20000, round(w * h * fps * 0.2 / 1000))


def encoder_flags(encoder: str, live: bool) -> list[str]:
    """Quality settings of each encoder: the best quality that still keeps up with a live stream. ``live``: a
    stream to a platform (constant bitrate, low delay); else a recording (variable bitrate, slower = better).
    ``ENCODER_PLAIN=1`` (the app's encoder test found these settings refused on this PC): none."""
    if os.environ.get("ENCODER_PLAIN") == "1":
        return []
    if encoder == "h264_nvenc":
        return ["-preset", "p5", "-tune", "hq", "-rc", "cbr" if live else "vbr", "-profile:v", "high", "-spatial-aq", "1"]
    if encoder == "h264_amf":
        return ["-quality", "quality", "-rc", "cbr" if live else "vbr_peak", "-profile:v", "high"]
    if encoder == "h264_qsv":
        return ["-preset", "slow", "-profile:v", "high"]
    if encoder == "libx264":
        return ["-preset", "veryfast", "-tune", "zerolatency"] if live else ["-preset", "fast", "-profile:v", "high"]
    return []


class FFmpegSink:
    """Raw BGR frames (+ their sound) -> FFmpeg -> H.264 (+ AAC) at ``url``. The container follows the URL scheme.

    ``audio`` (default): the output always has a sound track - Twitch and YouTube ingest need one - made of each
    frame's own samples, or silence for a frame written without sound. One frame and exactly one frame's worth of
    samples go down one pipe together, so the two cannot drift apart. False: picture only, as before.

    The connection is made when the first frame is written, not before: a platform never sees an idle connection
    while the models load or OBS has not started. For a network address the link is watched and said in
    ``STATUS`` lines (``io/link.py``):
      * it never came up (wrong address / key): tried ``GIVE_UP_BEFORE_FIRST`` times, then ``PlatformRefused``
        ends the run with the reason - nothing is retried for ever behind the streamer's back;
      * it was up and broke (network drop, the platform restarted): tried again for as long as the run lasts,
        after 1, 2, 4, 8, then every 15 seconds. Frames of the time in between are not sent at all, and the first
        second after a new connection is the black slate without sound: what viewers get around a drop is
        nothing or the slate, never a frame that skipped the protection (every frame written here already went
        through the gate).
    FFmpeg's own messages are passed on with the address and the key blanked out."""

    RETRY_S = (1.0, 2.0, 4.0, 8.0, 15.0)
    GIVE_UP_BEFORE_FIRST = 3
    IO_TIMEOUT_S = 10  # a network write that hangs this long is an error (FFmpeg -rw_timeout), not a frozen stream

    def __init__(
        self,
        url: str,
        size: tuple[int, int],
        fps: float = 30.0,
        encoder: str | None = None,  # None: $ENCODER (the installed app picks the one that works, app/encoders.py) or libx264
        bitrate: str | None = None,  # None: $VIDEO_KBPS (the app: the platform's figure) or auto_kbps()
        ffmpeg: str = "ffmpeg",
        audio: bool = True,
        audio_bitrate: str = "160k",
        spawn: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        exe = shutil.which(ffmpeg)
        if exe is None and spawn is None:
            raise FileNotFoundError("ffmpeg not found on PATH")
        encoder = encoder or os.environ.get("ENCODER") or "libx264"
        w, h = size
        self.size, self.fps = size, fps
        self.takes_audio = audio
        self.network = url.startswith(STREAM_SCHEMES)
        self.connected = False  # the platform is taking the stream right now
        self.reconnects = 0
        self.unsent = 0  # frames not sent because the link was down
        fmt = "flv" if url.startswith(("rtmp://", "rtmps://")) else "mpegts" if url.startswith(("srt://", "udp://")) else None
        cmd = [exe or ffmpeg, "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:2", "-stats_period", "0.5", "-y"]
        if audio:  # our own stream, nothing to probe: start at once
            cmd += ["-probesize", "32", "-analyzeduration", "0", "-f", "avi", "-i", "pipe:0"]
        else:
            cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0"]
        if bitrate is None:
            env_kbps = os.environ.get("VIDEO_KBPS", "")
            bitrate = f"{env_kbps}k" if env_kbps.isdigit() else f"{auto_kbps(size, fps, self.network)}k"
        self.bitrate = bitrate
        live = self.network
        kbps = int(bitrate.rstrip("kK")) if bitrate.rstrip("kK").isdigit() else 6000
        # live: constant rate (what platforms want); a file: variable, peaks up to 1.5x
        peak, buf = (kbps, kbps) if live else (round(kbps * 1.5), kbps * 2)
        cmd += ["-c:v", encoder, *encoder_flags(encoder, live), "-b:v", bitrate, "-maxrate", f"{peak}k", "-bufsize", f"{buf}k",
                "-g", str(round(2 * fps)), "-pix_fmt", "yuv420p"]  # fmt: skip
        if audio:
            cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", str(AUDIO_RATE), "-ac", "2"]
        if fmt:
            cmd += ["-f", fmt]
        if url.startswith(("rtmp://", "rtmps://")):
            cmd += ["-rw_timeout", str(self.IO_TIMEOUT_S * 1_000_000)]
        self._cmd = [*cmd, url]
        self._spawn = spawn or subprocess.Popen
        self._clock = clock
        self._secrets = link.Secrets(url if self.network else "")
        self._proc: Any = None
        self._avi: AviWriter | None = None
        self._errors: collections.deque[str] = collections.deque(maxlen=6)
        self._ever = False  # the platform took the stream at least once
        self._attempt = 0  # failed tries in a row
        self._retry_at = 0.0
        self._slate_left = 0
        self._sizes: list[int] = []
        self._closed = False

    # --- one FFmpeg process ---------------------------------------------------------------------------------
    def _open(self) -> None:
        self._errors.clear()
        self._sizes = []
        if self.network:
            link.status("platform_connecting")
        self._proc = self._spawn(self._cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._watch, args=(self._proc,), name="sink-output", daemon=True).start()
        try:
            self._avi = AviWriter(self._proc.stdin, self.size, self.fps) if self.takes_audio else None
        except OSError:
            self._lost()

    def _watch(self, proc: Any) -> None:
        """FFmpeg's progress (bytes that left for the platform) and its error lines, on their own thread."""
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return
        with contextlib.suppress(OSError, ValueError):
            for raw in stream:
                line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                m = _PROGRESS.match(line)
                if m is None:
                    if line:
                        clean = self._secrets.clean(line)
                        self._errors.append(clean)
                        print(f"OUTPUT {clean}", file=sys.stderr, flush=True)
                    continue
                if m.group(1) == "total_size" and m.group(2).strip().isdigit() and proc is self._proc:
                    self._progress(int(m.group(2)))

    def _progress(self, total: int) -> None:
        # connected = bytes kept leaving over two reports in a row (a refused stream dies before the second)
        if total > 0 and (not self._sizes or total > self._sizes[-1]):
            self._sizes.append(total)
        if len(self._sizes) >= 2 and not self.connected:
            self.connected, self._ever, self._attempt = True, True, 0
            if self.network:
                link.status("platform_connected")

    def _kill(self) -> None:
        proc, self._proc, self._avi = self._proc, None, None
        if proc is None:
            return
        with contextlib.suppress(OSError, ValueError):
            proc.stdin.close()
        try:
            proc.wait(timeout=3)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()

    def _lost(self) -> None:
        """This FFmpeg is gone. A file output: an error. A network output: try again later, or give up when the
        platform never took the stream."""
        self._kill()
        time.sleep(0.05)  # let the watcher read FFmpeg's last words
        detail = " | ".join(self._errors)[-300:]
        reason = link.reason_of(detail, "broken")
        was_up, self.connected = self.connected, False
        if not self.network:
            raise ConnectionError(f"ffmpeg output stopped {detail}".strip())
        self._attempt += 1
        if not self._ever and self._attempt >= self.GIVE_UP_BEFORE_FIRST:
            link.status("platform_failed", reason=reason, detail=detail)
            raise PlatformRefused(reason, detail)
        wait = self.RETRY_S[min(self._attempt, len(self.RETRY_S)) - 1]
        self._retry_at = self._clock() + wait
        self.reconnects += int(was_up)
        link.status("platform_lost", reason=reason, retry_s=wait, attempt=self._attempt, detail=detail)

    # --- frames ---------------------------------------------------------------------------------------------
    def write(self, frame: np.ndarray, audio: np.ndarray | None = None) -> None:
        if self._proc is None:
            if self._clock() < self._retry_at:
                self.unsent += 1
                return  # the link is down: this frame is simply not sent
            again = self._attempt > 0
            self._open()
            if self._proc is None:
                return
            if again:
                self._slate_left = int(self.fps)  # back on the air with the slate first
        elif self._proc.poll() is not None:
            self._lost()
            self.unsent += 1
            return
        if self._slate_left > 0:
            self._slate_left -= 1
            frame, audio = np.zeros_like(frame), None
        try:
            if self._avi is not None:
                self._avi.write(frame, audio)
            else:
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (OSError, ValueError):  # BrokenPipeError; on Windows a closed pipe is "Invalid argument"
            self.unsent += 1
            self._lost()

    def close(self) -> None:
        self._closed = True
        proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(OSError, ValueError):
            proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


class RecordSink(FFmpegSink):
    """The protected picture with its sound into a video file ("record only": a creator who uploads videos and
    does not stream). Written as Matroska while it runs - that file stays playable whatever happens (a crash, a
    power cut) - and turned into the MP4 that was asked for when the recording ends, without encoding again. If
    that last step cannot be done the .mkv stays and is the recording."""

    def __init__(self, path: str, size: tuple[int, int], fps: float = 30.0, bitrate: str | None = None, **kw: Any) -> None:
        kw.setdefault("audio_bitrate", "320k")  # a recording keeps the sound as good as it came
        self.path = str(path)
        root, ext = os.path.splitext(self.path)
        self._wants_mp4 = ext.lower() in (".mp4", ".mov", ".m4v")
        self._work = root + ".recording.mkv" if self._wants_mp4 else self.path
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        super().__init__(self._work, size, fps, bitrate=bitrate, **kw)
        self.result = ""

    def _open(self) -> None:
        link.status("recording", file=self.path)
        super()._open()

    def close(self) -> None:
        started = self._proc is not None
        super().close()
        if not started and not os.path.exists(self._work):
            return
        self.result = self._work
        if self._wants_mp4:
            cmd = [
                self._cmd[0],
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                self._work,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                self.path,
            ]
            try:
                done = self._spawn(cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if done.wait(timeout=600) == 0 and os.path.getsize(self.path) > 0:
                    os.remove(self._work)
                    self.result = self.path
            except (OSError, subprocess.SubprocessError):
                pass
        link.status("recorded", file=self.result)


class WindowSink:
    """The window OBS captures (Window Capture "StreamSafe Output"). Needs an OpenCV build with GUI support
    (``opencv-python``, not ``-headless``).

    The window lives on its own thread, which draws it about every 15 ms whatever the engine is doing: it is black
    (fail closed) from the moment it opens, while the models load or the input has not started, and whenever no new
    frame came for ``STALE_S``. So Windows never marks it "Not Responding" and OBS never captures a grey or frozen
    picture. Fixed size (the output frame, never scaled by dragging). Windows stops drawing a minimised window and
    OBS then captures nothing, so on Windows a minimised window is restored at once without taking the focus
    (``restored`` counts it); the OBS guard also shows the slate whenever the captured window is empty."""

    CHECK_S = 0.25
    STALE_S = 0.5  # no new frame for this long: the window goes black instead of freezing on the last picture
    TICK_MS = 15
    IDLE_SIZE = (1280, 720)

    def __init__(self, title: str = WINDOW_TITLE) -> None:
        self.title = title
        self.restored = 0
        self.shown = 0  # frames from the engine that were drawn
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._at = 0.0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="window", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self._error is not None:
            raise self._error

    def _black(self, like: np.ndarray | None) -> np.ndarray:
        if like is not None:
            return np.zeros_like(like)
        return np.zeros((self.IDLE_SIZE[1], self.IDLE_SIZE[0], 3), np.uint8)

    def _loop(self) -> None:
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self.title, self._black(None))
            cv2.waitKey(1)
        except BaseException as exc:  # no GUI support: the caller gets the error
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        next_check, last = 0.0, None
        while not self._stop.is_set():
            with self._lock:
                frame, at = self._frame, self._at
                self._frame = None
            now = time.monotonic()
            if frame is not None:
                last = frame
                cv2.imshow(self.title, frame)
                self.shown += 1
            elif last is not None and now - at > self.STALE_S:
                cv2.imshow(self.title, self._black(last))  # stale: fail closed, black at the same size
                last = None
            cv2.waitKey(self.TICK_MS)
            if sys.platform == "win32" and now >= next_check:
                next_check = now + self.CHECK_S
                self.restored += _restore_if_minimised(self.title)
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)

    def write(self, frame: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        with self._lock:
            self._frame = frame
            self._at = time.monotonic()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2)


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
    frame after writing it (the engine never does - each output frame is a fresh buffer).

    ``keepalive_fps`` (a stream to a platform): once the first frame went out, a gap of more than ``STALE_S``
    without a new frame (OBS stopped sending, the input is being reopened) is filled with the black slate and
    silence at that rate - the platform keeps a live stream that shows the slate instead of dropping the channel
    or freezing on the last picture. ``filled`` counts those frames."""

    STALE_S = 0.5

    def __init__(self, sink: FrameSink, max_queue: int = 64, keepalive_fps: float = 0.0) -> None:
        self.sink = sink
        self.takes_audio = takes_audio(sink)
        self.keepalive_fps = keepalive_fps
        self.filled = 0
        self._queue: queue.Queue[tuple[np.ndarray, np.ndarray | None] | None] = queue.Queue(maxsize=max_queue)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="sink", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last: np.ndarray | None = None
        wait: float | None = None
        while True:
            try:
                item = self._queue.get(timeout=wait)
            except queue.Empty:
                if self._error is None and last is not None:
                    try:
                        _write(self.sink, np.zeros_like(last), None)
                        self.filled += 1
                    except BaseException as exc:
                        self._error = exc
                wait = 1.0 / self.keepalive_fps
                continue
            if item is None:
                return
            if self._error is None:
                try:
                    _write(self.sink, *item)
                except BaseException as exc:  # reported to the caller on the next write
                    self._error = exc
            if self.keepalive_fps > 0:
                last, wait = item[0], self.STALE_S

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
    ``record:<file>`` -> a full-quality recording with sound (``RecordSink``); else an MP4 review file (picture only)."""
    if spec == "window":
        return WindowSink()
    if spec.startswith("record:"):
        return ThreadedSink(RecordSink(spec[len("record:") :], size, fps, audio=audio))
    if "://" in spec:
        return ThreadedSink(FFmpegSink(spec, size, fps, audio=audio), keepalive_fps=fps if spec.startswith(STREAM_SCHEMES) else 0.0)
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
