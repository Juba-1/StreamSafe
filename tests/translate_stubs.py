"""Stand-ins for the translation models, and synthetic frames with English subtitles (top-level so the translation
worker process can import them)."""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from streamsafe.config import Config
from streamsafe.translate import pipeline as pl
from streamsafe.translate.arabic import ArabicRenderer
from streamsafe.translate.ocr import Line
from streamsafe.translate.service import REGIONS, region_rect

SIZE = (1920, 1080)
LINES = {  # English line -> its Arabic stand-in
    "Follow me to the private room.": "اتبعني إلى الغرفة الخاصة.",
    "Press X to end the dance": "اضغط X لإنهاء الرقص",
    "Where have you been all night, Lucia?": "أين كنت طوال الليل يا لوسيا؟",
}
BG = (90, 120, 140)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text_width(text: str, scale: float = 1.4) -> int:
    return cv2.getTextSize(text, FONT, scale, 3)[0][0]


def draw_subtitle(frame: np.ndarray, text: str, scale: float = 1.4) -> tuple[int, int, int, int]:
    """White subtitle with a dark outline, centred at the bottom like the game's. Returns its box (x0, y0, x1, y1)."""
    w, h = frame.shape[1], frame.shape[0]
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 3)
    x, y = (w - tw) // 2, int(h * 0.9)
    cv2.putText(frame, text, (x, y), FONT, scale, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, (255, 255, 255), 3, cv2.LINE_AA)
    return x - 4, y - th - 4, x + tw + 4, y + base + 4


def subtitle_frames(n: int, shows: dict[tuple[int, int], str], size: tuple[int, int] = SIZE, person: bool = True) -> list[np.ndarray]:
    """``n`` frames: a plain background, a walking red person (so the gate has something to cover), and the subtitle
    ``shows[(a, b)]`` on frames a <= f < b."""
    frames = []
    w, h = size
    for f in range(n):
        im = np.full((h, w, 3), BG, np.uint8)
        if person:
            cv2.ellipse(im, (int((400 + 6 * f) * w / 1920), int(520 * h / 1080)), (60, 190), 0, 0, 360, (40, 40, 200), -1)
        for (a, b), text in shows.items():
            if a <= f < b:
                draw_subtitle(im, text)
        frames.append(im)
    return frames


class ListSource:
    """A paced (file-like) source made of frames in memory."""

    live = False

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self.size = (frames[0].shape[1], frames[0].shape[0])

    def frames(self):
        yield from self._frames

    def close(self) -> None:
        pass


class LazySource:
    """A paced source whose frames are drawn one at a time (long runs would not fit in memory as a list)."""

    live = False

    def __init__(self, n: int, make, size: tuple[int, int] = SIZE) -> None:
        self._n, self._make, self.size = n, make, size

    def frames(self):
        for f in range(self._n):
            yield self._make(f)

    def close(self) -> None:
        pass


class RecordingSink:
    """Keeps a hash of every output frame (exact comparison between runs)."""

    def __init__(self) -> None:
        import hashlib

        self._hash = hashlib.sha1
        self.hashes: list[str] = []
        self.frames: list[np.ndarray] = []
        self.keep_frames = False

    def write(self, frame: np.ndarray) -> None:
        self.hashes.append(self._hash(frame.tobytes()).hexdigest())
        if self.keep_frames:
            self.frames.append(frame.copy())

    def close(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------ stub models
class StubReader:
    """Finds the white subtitle (bright pixels) and knows the English line by its width."""

    def read(self, bgr: np.ndarray) -> list[Line]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray > 200).astype(np.uint8)
        if int(mask.sum()) < 30:
            return []
        x, y, w, h = cv2.boundingRect(mask)
        ratio = w / (bgr.shape[1] - 2 * pl.EDGE)
        region_w = region_rect("subtitles", SIZE)[2] - region_rect("subtitles", SIZE)[0]
        best = min(LINES, key=lambda t: abs(text_width(t) / region_w - ratio))
        return [Line((x - 3, y - 3, x + w + 3, y + h + 3), best, 0.99)]


class StubTranslator:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay, self.calls = delay, []

    def translate(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        time.sleep(self.delay)
        return [LINES.get(t, "ترجمة") for t in texts]


def _pipeline(translator) -> pl.Pipeline:
    return pl.Pipeline(StubReader(), translator, ArabicRenderer(None))


def stub_factory(cfg: Config) -> pl.Pipeline:
    return _pipeline(StubTranslator())


def slow_factory(cfg: Config) -> pl.Pipeline:
    """Every read takes 5 s: nothing is ready in time (the frames wait 1 s)."""
    return _pipeline(StubTranslator(delay=5.0))


class _Crashing(StubTranslator):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def translate(self, texts: list[str]) -> list[str]:
        if not self.calls:  # the warm-up call at load time works; the first real line fails
            self.calls.append(list(texts))
            return ["مرحبا"]
        if self.mode == "exit":
            os._exit(3)  # the whole process dies, like a crash inside a model library
        raise RuntimeError("translator failure")


def dying_factory(cfg: Config) -> pl.Pipeline:
    return _pipeline(_Crashing("exit"))


def failing_factory(cfg: Config) -> pl.Pipeline:
    return _pipeline(_Crashing("raise"))


def broken_load_factory(cfg: Config) -> pl.Pipeline:
    raise RuntimeError("models are damaged")


def all_regions() -> list[str]:
    return list(REGIONS)
