"""Shows the translation on a made-up game frame with the real models: writes a before / after picture and prints
what was read, translated and how long each step took.

    python tools/translate_demo.py PACK_DIR OUT.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

LINES = ("Follow me to the private room.", "Where have you been all night, Lucia? I was worried sick.", "Press X to end the dance")


def find_font() -> str:
    from PIL import ImageFont

    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            ImageFont.truetype(name, 20)
            return name
        except OSError:
            continue
    raise SystemExit("no TrueType font found")


def scene(size=(1920, 1080)) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):  # a dusk gradient with a few shapes, so the text sits on something that is not flat
        img[y] = (int(60 + 80 * y / h), int(70 + 40 * y / h), int(110 - 50 * y / h))
    cv2.rectangle(img, (int(w * 0.1), int(h * 0.55)), (int(w * 0.35), int(h * 0.9)), (50, 55, 60), -1)
    cv2.ellipse(img, (int(w * 0.62), int(h * 0.55)), (70, 190), 0, 0, 360, (60, 70, 160), -1)
    return img


def subtitle(img: np.ndarray, text: str, font: str, px: int = 44, y: float = 0.9) -> np.ndarray:
    """White game-style subtitle, black outline, centred; wrapped at 80 % of the width."""
    from PIL import Image, ImageDraw, ImageFont

    f = ImageFont.truetype(font, px)
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if cur and f.getlength(t) > 0.62 * img.shape[1]:
            lines.append(cur)
            cur = wd
        else:
            cur = t
    lines.append(cur)
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    top = int(img.shape[0] * y) - (len(lines) - 1) * int(px * 1.25)
    for i, ln in enumerate(lines):
        x, yy = (img.shape[1] - f.getlength(ln)) / 2, top + i * int(px * 1.25)
        d.text((x, yy), ln, font=f, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    return np.asarray(pil)[:, :, ::-1].copy()


def main(pack_dir: str, out: str) -> dict:
    from streamsafe.translate.arabic import blend
    from streamsafe.translate.pipeline import load_pipeline
    from streamsafe.translate.service import region_rect

    t0 = time.perf_counter()
    pipe = load_pipeline(Path(pack_dir))
    pipe.warmup()
    load_s = time.perf_counter() - t0
    font = find_font()
    rows, report = [], {"load_s": round(load_s, 2), "lines": []}
    size = (1920, 1080)
    rect = region_rect("subtitles", size)
    for text in LINES:
        before = subtitle(scene(size), text, font)
        crop = before[rect[1] : rect[3], rect[0] : rect[2]].copy()
        t = time.perf_counter()
        got = pipe({"subtitles": (crop, rect)}, size)
        ms = 1000 * (time.perf_counter() - t)
        after = before.copy()
        for p in got["subtitles"]:
            blend(after, p)
        report["lines"].append(dict(english=text, patches=len(got["subtitles"]), ms=round(ms, 1)))
        rows.append(np.vstack([before[int(1080 * 0.62) :], after[int(1080 * 0.62) :]]))
    sheet = np.vstack(rows)
    cv2.imwrite(out, cv2.resize(sheet, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
    report["worker_ms"] = {k: [round(x, 1) for x in v] for k, v in pipe.ms.items()}
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
