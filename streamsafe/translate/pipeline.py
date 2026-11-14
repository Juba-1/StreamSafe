"""The worker's job: from the picture of a screen region to finished Arabic boxes.

Runs inside the translation process (``service.py``). Steps per region: read the English lines, keep what is
confidently English text, join lines of one subtitle into a paragraph, translate it (cached), render the Arabic
box that covers the English paragraph. A region whose English is the same as the last time it was read is not
translated or drawn again (``SAME``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from streamsafe.translate import textproc
from streamsafe.translate.arabic import ArabicRenderer, Patch
from streamsafe.translate.translator import Translator

READ_W = 1280  # a region is read at this width at most (text stays readable, cost bounded)
EDGE = 24  # black border around the region: OCR misses text that touches the image edge
SAME = "same"

Crop = tuple[np.ndarray, tuple[int, int, int, int]]  # BGR region cut from the frame, its (x0, y0, x1, y1) in the frame


@dataclass
class Paragraph:
    box: tuple[int, int, int, int]  # crop pixels
    text: str
    lines: int = 1
    conf: float = 1.0


def group(lines, gap: float = 0.9) -> list[Paragraph]:
    """Lines of one subtitle (close together, overlapping horizontally) -> one paragraph each."""
    paras: list[Paragraph] = []
    for ln in sorted(lines, key=lambda ln: ln.box[1]):
        x0, y0, x1, y1 = ln.box
        for p in paras:
            px0, py0, px1, py1 = p.box
            overlap = min(x1, px1) - max(x0, px0)
            if y0 - py1 <= gap * (y1 - y0) and overlap > 0.2 * min(x1 - x0, px1 - px0):
                p.box = (min(px0, x0), py0, max(px1, x1), max(py1, y1))
                p.text += " " + ln.text
                p.lines += 1
                p.conf = min(p.conf, ln.conf)
                break
        else:
            paras.append(Paragraph(ln.box, ln.text, 1, ln.conf))
    return paras


@dataclass
class Pipeline:
    reader: Any  # TextReader (or a stand-in with .read(bgr) -> [Line])
    translator: Translator
    renderer: ArabicRenderer
    min_conf: float = 0.55
    _last: dict[str, list[str]] = field(default_factory=dict)
    ms: dict[str, list[float]] = field(default_factory=lambda: {"ocr": [], "translate": [], "render": []})

    def warmup(self) -> None:
        """Run every stage once on a made-up line, so the first real subtitle does not pay for the models' first call
        (lazy allocations, font loading): on a slow CPU that was seconds, longer than the stream's delay."""
        self.translator.translate(["Press X to end the dance."])
        img = np.zeros((96, 640, 3), np.uint8)
        cv2.putText(img, "Press X to end the dance", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        try:  # best effort: a stage that cannot run on this picture is not a reason to refuse to start
            self.reader.read(cv2.copyMakeBorder(img, EDGE, EDGE, EDGE, EDGE, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
            self.renderer.render("اضغط X لإنهاء الرقص", (100, 400, 500, 450), (1280, 720), 1)
        except Exception:
            pass
        for v in self.ms.values():
            v.clear()

    def __call__(self, crops: dict[str, Crop | None], frame_size: tuple[int, int]) -> dict[str, Any]:
        """Per region name: a list of ``Patch`` (possibly empty = no English text there), or ``SAME``. A region whose
        crop is ``None`` is not read: its text has not changed (``SAME``). A region that is not given has no text."""
        out: dict[str, Any] = {}
        for gone in [n for n in self._last if n not in crops]:
            del self._last[gone]  # no text there any more: when text comes back it is read and drawn again
        for name, item in crops.items():
            if item is None:
                out[name] = SAME
                continue
            bgr, rect = item
            h, w = bgr.shape[:2]
            scale = min(1.0, READ_W / w)
            small = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
            padded = cv2.copyMakeBorder(small, EDGE, EDGE, EDGE, EDGE, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            t = time.perf_counter()
            lines = [
                ln for ln in self.reader.read(padded) if ln.conf >= self.min_conf and textproc.worth_translating(textproc.clean(ln.text))
            ]
            self.ms["ocr"].append(1000 * (time.perf_counter() - t))
            paras = [p for p in group(lines) if textproc.worth_translating(p.text)]
            keys = [textproc.cache_key(p.text) for p in paras]
            if name in self._last and self._last[name] == keys:
                out[name] = SAME
                continue
            self._last[name] = keys
            t = time.perf_counter()
            arabic = self.translator.translate([textproc.clean(p.text) for p in paras]) if paras else []
            self.ms["translate"].append(1000 * (time.perf_counter() - t))
            t = time.perf_counter()
            patches: list[Patch] = []
            for p, ar in zip(paras, arabic, strict=True):
                if not ar:
                    continue
                x0, y0, x1, y1 = p.box  # padded-small pixels -> frame pixels
                box = (
                    rect[0] + round((x0 - EDGE) / scale),
                    rect[1] + round((y0 - EDGE) / scale),
                    rect[0] + round((x1 - EDGE) / scale),
                    rect[1] + round((y1 - EDGE) / scale),
                )
                patch = self.renderer.render(ar, box, frame_size, p.lines)
                if patch is not None:
                    patches.append(patch)
            self.ms["render"].append(1000 * (time.perf_counter() - t))
            out[name] = patches
        return out


def load_pipeline(folder: Path, threads: int | None = None, min_conf: float = 0.55) -> Pipeline:
    """The real models from an installed pack."""
    if threads is None:  # leave the cores to the game and the stream: 1 thread on a small CPU, 2 from 8 cores up
        threads = 2 if (os.cpu_count() or 2) >= 8 else 1
    from streamsafe.translate.ocr import TextReader
    from streamsafe.translate.translator import CachedTranslator, Ct2Translator, GameTranslator

    return Pipeline(
        TextReader(folder, threads), CachedTranslator(GameTranslator(Ct2Translator(folder, threads))), ArabicRenderer(folder), min_conf
    )
