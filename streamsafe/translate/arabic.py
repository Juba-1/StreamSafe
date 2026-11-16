"""Arabic on a picture: letter shaping, right-to-left order, line breaking and a readable box.

Shaping (isolated / initial / medial / final forms, lam-alef) is ``arabic-reshaper`` (MIT). The right-to-left
ordering is the Unicode bidirectional algorithm, written in ``bidi.py`` (the usual ``python-bidi`` is LGPL and
Pillow's own bidi needs libraries Windows wheels do not carry). Text is drawn with Pillow's BASIC layout engine so
the result is the same on every platform.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from streamsafe.translate import bidi, textproc

FONT_FILE = "NotoSansArabic-Regular.ttf"  # SIL OFL 1.1; the pack carries both fonts (fallbacks below for a missing file)
LATIN_FILE = "NotoSans-Regular.ttf"  # Noto Sans Arabic has no Latin letters: names, buttons ("X") and digits use this
_FALLBACK_FONTS = ("segoeui.ttf", "tahoma.ttf", "arial.ttf", "DejaVuSans.ttf", "NotoSansArabic-Regular.ttf")
_FALLBACK_LATIN = ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf")
BG = (20, 20, 20)  # BGR of the box: near black, like the dark outline game subtitles are drawn with
BG_ALPHA = 255  # fully opaque: not one letter of the English may show through (owner requirement)
PAD_X_K, PAD_Y_K = 0.45, 0.35  # padding around the English text, as a share of its line height
MAX_WIDTH = 0.92  # of the frame width


def visual_order(text: str, base: str | None = None) -> str:
    """Logical text -> the characters in the order they are drawn from left to right: the Unicode bidirectional
    algorithm (``bidi.reorder``). ``base`` 'R' / 'L' is the direction of the paragraph; None: right-to-left when the
    text has an Arabic letter (the presentation forms the shaper writes are Arabic letters too), else left-to-right."""
    return bidi.reorder(text, base)


_PLUS = re.compile(r"\b[A-Za-z]+[+#]+(?=\s|$)")  # C++, C#


def hold_together(text: str) -> str:
    """Marks around what must stay one left-to-right piece inside the Arabic line: "@name" and "#tag" (the bare
    algorithm puts the sign on the far side of the name: "name@"), web addresses, player and brand names with signs
    in them ("WhatUp!"), "C++". Only for a line that has Arabic: a Latin line needs none."""
    if bidi.base_of(text) != "R":
        return text
    return _PLUS.sub(lambda m: bidi.ltr(m.group(0)), textproc.sub_keep(text, bidi.ltr))


_reshaper = None


def shape(text: str, base: str | None = None) -> str:
    """Logical Arabic -> shaped letters in drawing order (left to right). ``base``: the paragraph's direction when
    ``text`` is one line of a longer paragraph (a line of Latin names inside an Arabic paragraph is still read from
    the right)."""
    global _reshaper
    if _reshaper is None:
        import arabic_reshaper

        _reshaper = arabic_reshaper.ArabicReshaper(configuration={"delete_harakat": True, "support_ligatures": True})
    return visual_order(hold_together(_reshaper.reshape(text)), base)


@dataclass(frozen=True)
class Patch:
    """A finished picture to blend onto the output: ``bgr`` colours, ``alpha`` 0..255 per pixel, top-left (x, y)."""

    x: int
    y: int
    bgr: np.ndarray
    alpha: np.ndarray


class ArabicRenderer:
    def __init__(self, font_dir: Path | None = None) -> None:
        self.font_dir = font_dir
        self._fonts: dict[tuple[str, int], object] = {}
        self._path = self._find(FONT_FILE, _FALLBACK_FONTS)
        self._latin = self._find(LATIN_FILE, _FALLBACK_LATIN, required=False) or self._path

    def _find(self, name: str, fallbacks: tuple[str, ...], required: bool = True) -> str:
        from PIL import ImageFont

        first = [str(self.font_dir / name)] if self.font_dir else []
        for cand in (*first, *fallbacks):
            try:
                ImageFont.truetype(cand, 20)
                return cand
            except OSError:
                continue
        if required:
            raise RuntimeError("no font with Arabic letters found")
        return ""

    def font(self, px: int, latin: bool = False):
        from PIL import ImageFont

        key = ("l" if latin else "a", px)
        f = self._fonts.get(key)
        if f is None:
            f = self._fonts[key] = ImageFont.truetype(self._latin if latin else self._path, px, layout_engine=ImageFont.Layout.BASIC)
        return f

    @staticmethod
    def _segments(visual: str) -> list[tuple[bool, str]]:
        """The drawn string cut into runs by font: (is_latin, text). ASCII (letters, digits, ``( ) %``) is Latin."""
        out: list[tuple[bool, str]] = []
        for c in visual:
            latin = c.isascii() and not c.isspace()
            if out and (out[-1][0] == latin or c.isspace()):
                out[-1] = (out[-1][0], out[-1][1] + c)
            else:
                out.append((latin, c))
        return out

    def _length(self, visual: str, px: int) -> float:
        return sum(self.font(px, latin).getlength(t) for latin, t in self._segments(visual))

    def _wrap(self, text: str, px: int, max_w: float, base: str | None = None) -> list[str]:
        lines: list[str] = []
        cur: list[str] = []
        for word in text.split():
            trial = " ".join([*cur, word])
            if cur and self._length(shape(trial, base), px) > max_w:
                lines.append(" ".join(cur))
                cur = [word]
            else:
                cur.append(word)
        if cur:
            lines.append(" ".join(cur))
        return lines

    def render(self, text: str, box: tuple[int, int, int, int], frame_size: tuple[int, int], lines: int = 1) -> Patch | None:
        """A box of Arabic that covers the English text in ``box`` (x0, y0, x1, y1, frame pixels): the same
        centre, at least as wide and as tall, anchored at the English text's bottom edge, white text on a dark
        opaque ground. Wider (up to 92 % of the frame) or taller (upwards) when the Arabic needs room."""
        from PIL import Image, ImageDraw

        fw, fh = frame_size
        x0, y0, x1, y1 = box
        en_w, en_h = x1 - x0, y1 - y0
        if not text.strip() or en_w < 8 or en_h < 8:
            return None
        px = int(min(max(en_h / max(1, lines) * 0.95, 20), 0.07 * fh))  # about as tall as one line of the English
        pad_x, pad_y = round(PAD_X_K * en_h), round(PAD_Y_K * en_h)
        max_w = min(MAX_WIDTH * fw, max(en_w + 2 * pad_x, 0.5 * fw)) - 2 * pad_x
        base = bidi.base_of(text)  # of the whole paragraph: every line of it is laid out in that direction
        lines = self._wrap(text, px, max_w, base)
        shaped = [shape(ln, base) for ln in lines]
        text_w = max(self._length(s, px) for s in shaped)
        line_h = round(px * 1.45)
        w = int(min(MAX_WIDTH * fw, max(en_w + 2 * pad_x, text_w + 2 * pad_x)))
        h = int(max(en_h + 2 * pad_y, len(lines) * line_h + 2 * pad_y))
        cx = (x0 + x1) / 2
        px0 = int(min(max(0, round(cx - w / 2)), fw - w))
        bottom = min(fh, y1 + pad_y)
        py0 = max(0, bottom - h)
        h = min(h, fh - py0)
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        top = (h - len(lines) * line_h) / 2
        for i, s in enumerate(shaped):
            x = (w - self._length(s, px)) / 2
            base = top + i * line_h + line_h / 2 + px * 0.34  # baseline of this line
            for latin, seg in self._segments(s):
                f = self.font(px, latin)
                draw.text((x, base), seg, font=f, fill=255, anchor="ls")
                x += f.getlength(seg)
        t = np.asarray(mask, np.float32) / 255.0
        bg = np.empty((h, w, 3), np.float32)
        bg[:] = BG
        bgr = (bg * (1 - t[..., None]) + 255.0 * t[..., None]).astype(np.uint8)
        alpha = (BG_ALPHA + (255 - BG_ALPHA) * t).astype(np.uint8)
        return Patch(px0, py0, bgr, alpha)


def blend(frame: np.ndarray, patch: Patch) -> None:
    """Blend ``patch`` into ``frame`` (BGR uint8, in place), clipped to the frame. Touches only the patch's area."""
    fh, fw = frame.shape[:2]
    h, w = patch.alpha.shape
    x0, y0 = max(0, patch.x), max(0, patch.y)
    x1, y1 = min(fw, patch.x + w), min(fh, patch.y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sy, sx = y0 - patch.y, x0 - patch.x
    bgr = patch.bgr[sy : sy + y1 - y0, sx : sx + x1 - x0].astype(np.uint16)
    a = patch.alpha[sy : sy + y1 - y0, sx : sx + x1 - x0].astype(np.uint16)[..., None]
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = ((roi.astype(np.uint16) * (255 - a) + bgr * a + 127) // 255).astype(np.uint8)
