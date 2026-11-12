"""Picture and sound through ONE pipe to / from FFmpeg, as an endless AVI stream (raw BGR + 16-bit PCM).

Why one pipe: sound must stay glued to the picture. A second pipe (or socket) would need its own plumbing on
Windows and could dead-lock against the first; one ordered stream needs neither. AVI because it is the simplest
container FFmpeg both writes and reads on a pipe: a fixed header, then ``<id><size><data>`` chunks.

How sync is kept (no wall-clock guesses anywhere):
  * reading (``AviReader`` + ``AudioBuffer``): FFmpeg puts both streams on the sender's timestamps from one common
    zero (``fps=...:start_time=0`` for the picture, ``aresample=async:first_pts=0`` for the sound, see
    ``sources.py``), so frame ``k`` is the time ``[k/fps, (k+1)/fps)`` and its sound is exactly the samples
    ``sample_bounds(k)`` of the PCM stream. A frame and its sound are one unit from there on;
  * writing (``AviWriter``): every frame goes out with exactly one frame's worth of samples, so the encoder's
    two clocks (frames counted, samples counted) can never drift apart, whatever was dropped on the way.
"""

from __future__ import annotations

import struct
import threading
from typing import BinaryIO

import numpy as np

AUDIO_RATE = 48000  # Hz; what Twitch / Kick / YouTube ingest expect
AUDIO_CHANNELS = 2
SAMPLE_BYTES = 2 * AUDIO_CHANNELS  # s16le, interleaved
_CONTAINERS = (b"RIFF", b"LIST")


def sample_bounds(index: int, fps: float, rate: int = AUDIO_RATE) -> tuple[int, int]:
    """First and one-past-last sample of frame ``index``. Rounded from the running total, so the chunks tile the
    stream with no gap and no drift at any frame rate (30 fps: always 1600 samples)."""
    return round(index * rate / fps), round((index + 1) * rate / fps)


def row_bytes(width: int) -> int:
    """AVI stores raw RGB rows padded to 4 bytes."""
    return (width * 3 + 3) & ~3


# ------------------------------------------------------------------------------------------ reading
class AviReader:
    """Chunks of an AVI stream written by FFmpeg (``-f avi pipe:1``), in order.

    ``next()`` -> ("video", bytes) | ("audio", bytes) | None at the end of the stream. ``has_audio`` is known
    after the header (read on the first ``next()``): an input without sound simply has no audio stream."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._kinds: list[bytes] = []  # stream number -> b"vids" | b"auds"
        self._in_movi = False
        self.has_audio = False

    def _read(self, n: int) -> bytes | None:
        buf = self._stream.read(n)
        if len(buf) == n:
            return buf
        while buf and len(buf) < n:  # an unbuffered pipe may return less than asked
            more = self._stream.read(n - len(buf))
            if not more:
                break
            buf += more
        return buf if len(buf) == n else None

    def _skip(self, n: int) -> bool:
        while n > 0:
            part = self._stream.read(min(n, 1 << 20))
            if not part:
                return False
            n -= len(part)
        return True

    def _header(self) -> bool:
        """Reads up to the first data chunk; False if the stream ends before."""
        if self._read(12) is None:  # RIFF <size> AVI
            return False
        while True:
            head = self._read(8)
            if head is None:
                return False
            tag, size = head[:4], struct.unpack("<I", head[4:])[0]
            if tag == b"LIST":
                kind = self._read(4)
                if kind is None:
                    return False
                if kind == b"movi":
                    self._in_movi = True
                    self.has_audio = b"auds" in self._kinds
                    return True
                if kind not in (b"hdrl", b"strl") and not self._skip(size - 4 + (size & 1)):
                    return False
            elif tag == b"strh":
                data = self._read(size + (size & 1))
                if data is None:
                    return False
                self._kinds.append(data[:4])
            elif not self._skip(size + (size & 1)):
                return False

    def next(self) -> tuple[str, bytes] | None:
        if not self._in_movi and not self._header():
            return None
        while True:
            head = self._read(8)
            if head is None:
                return None
            tag, size = head[:4], struct.unpack("<I", head[4:])[0]
            if tag in _CONTAINERS:  # a new RIFF / LIST around more data chunks
                if self._read(4) is None:
                    return None
                continue
            number = tag[:2]
            if number.isdigit() and int(number) < len(self._kinds) and tag[2:] in (b"db", b"dc", b"wb"):
                data = self._read(size)
                if data is None or (size & 1 and self._read(1) is None):
                    return None
                return ("video" if self._kinds[int(number)] == b"vids" else "audio"), data
            if not self._skip(size + (size & 1)):  # index, padding, ...
                return None


class AudioBuffer:
    """The PCM stream of a source as it arrives, addressed by frame: ``chunk(k)`` is the sound of frame ``k``.

    Keeps ``keep_s`` seconds at most (older sound is forgotten, like frames that were never shown).
    Thread-safe: the reader thread appends, the output thread takes chunks."""

    def __init__(self, fps: float, keep_s: float = 30.0) -> None:
        self.fps = fps
        self.keep_s = keep_s
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._start = 0  # sample number of the first sample in _buf
        self._tail = b""  # an incomplete sample at the end of the last piece

    def append(self, pcm: bytes) -> None:
        with self._lock:
            if self._tail:
                pcm = self._tail + pcm
            whole = len(pcm) - len(pcm) % SAMPLE_BYTES
            self._tail = pcm[whole:]
            self._buf += pcm[:whole]
            extra = len(self._buf) // SAMPLE_BYTES - int(self.keep_s * AUDIO_RATE)
            if extra > AUDIO_RATE:  # trim a second at a time, not on every piece
                del self._buf[: extra * SAMPLE_BYTES]
                self._start += extra

    def chunk(self, index: int) -> np.ndarray | None:
        """Samples of frame ``index`` as int16 [n, channels]; None if they did not arrive (yet) or are gone."""
        a, b = sample_bounds(index, self.fps)
        with self._lock:
            if a < self._start or b > self._start + len(self._buf) // SAMPLE_BYTES:
                return None
            raw = bytes(self._buf[(a - self._start) * SAMPLE_BYTES : (b - self._start) * SAMPLE_BYTES])
        return np.frombuffer(raw, "<i2").reshape(-1, AUDIO_CHANNELS)


# ------------------------------------------------------------------------------------------ writing
def _chunk(tag: bytes, data: bytes) -> bytes:
    return tag + struct.pack("<I", len(data)) + data + (b"\0" if len(data) & 1 else b"")


def _list(kind: bytes, *parts: bytes) -> bytes:
    return _chunk(b"LIST", kind + b"".join(parts))


def _stream_header(kind: bytes, scale: int, rate: int, buffer: int, sample: int, w: int = 0, h: int = 0) -> bytes:
    packed = struct.pack("<4s4sIHHIIIIIIIIhhhh", kind, b"\0" * 4, 0, 0, 0, 0, scale, rate, 0, 0, buffer, 0xFFFFFFFF, sample, 0, 0, w, h)
    return _chunk(b"strh", packed)


def avi_header(size: tuple[int, int], fps: float) -> bytes:
    """Header of an endless AVI: stream 0 raw BGR (top-down), stream 1 PCM s16le stereo. Sizes that are not
    known for a live stream are 0, which FFmpeg reads as "until the pipe ends"."""
    w, h = size
    frame_bytes = row_bytes(w) * h
    byte_rate = AUDIO_RATE * SAMPLE_BYTES
    avih = struct.pack("<14I", round(1e6 / fps), 0, 0, 0, 0, 0, 2, frame_bytes, w, h, 0, 0, 0, 0)
    bmp = struct.pack("<IiiHHIIiiII", 40, w, -h, 1, 24, 0, frame_bytes, 0, 0, 0, 0)  # negative height: top-down
    wav = struct.pack("<HHIIHH", 1, AUDIO_CHANNELS, AUDIO_RATE, byte_rate, SAMPLE_BYTES, 16)
    hdrl = _list(
        b"hdrl",
        _chunk(b"avih", avih),
        _list(b"strl", _stream_header(b"vids", 1000, round(fps * 1000), frame_bytes, 0, w, h), _chunk(b"strf", bmp)),
        _list(b"strl", _stream_header(b"auds", 1, AUDIO_RATE, byte_rate, SAMPLE_BYTES), _chunk(b"strf", wav)),
    )
    return b"RIFF" + struct.pack("<I", 0) + b"AVI " + hdrl + b"LIST" + struct.pack("<I", 0) + b"movi"


class AviWriter:
    """Writes frames with their sound to a pipe FFmpeg reads with ``-f avi -i pipe:0``.

    Each ``write`` is one frame and exactly that frame's number of samples: ``audio`` (int16 [n, 2]) is cut or
    padded with silence to fit, None is silence. So picture and sound stay locked in the encoder."""

    def __init__(self, stream: BinaryIO, size: tuple[int, int], fps: float) -> None:
        self._stream, self.size, self.fps = stream, size, fps
        self._n = 0
        self._pad = row_bytes(size[0]) - size[0] * 3
        stream.write(avi_header(size, fps))

    def write(self, frame: np.ndarray, audio: np.ndarray | None = None) -> None:
        a, b = sample_bounds(self._n, self.fps)
        n = b - a
        pcm = np.zeros((n, AUDIO_CHANNELS), "<i2")
        if audio is not None and len(audio):
            m = min(n, len(audio))
            pcm[:m] = audio[:m]
        if self._pad:  # rows padded to 4 bytes (never for 16:9 sizes)
            rows = frame.reshape(frame.shape[0], -1)
            frame = np.concatenate([rows, np.zeros((len(rows), self._pad), np.uint8)], axis=1)
        video = np.ascontiguousarray(frame)
        # sound first: the encoder then has both for this instant as soon as the picture is in
        self._stream.write(b"01wb" + struct.pack("<I", pcm.nbytes))
        self._stream.write(pcm.tobytes())
        self._stream.write(b"00db" + struct.pack("<I", video.nbytes))
        self._stream.write(video.data)
        self._n += 1
