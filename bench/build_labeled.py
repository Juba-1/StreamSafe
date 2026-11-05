"""Embed every labeled game crop of the earlier rounds with the engine's own eye, keeping two labelings side by side.

Rounds: ``old`` + ``newcrops`` (label.html, labeled by Ali), video rounds 1-2 and clubs round 3 (contact sheets, labeled by
Claude; ``LABELS.json`` holds the original codes and the codes after Ali's review, per round and sheet). A crop is kept
when either labeling makes it R/M/S, so recipes trained on the old and on the corrected labels see the same rows.

usage: HEAD_DIR=<heads> python -m bench.build_labeled OUT.npz LABELS.json DATA_DIR NEWCROPS_DIR ROUND1_DIR ROUND2_DIR ROUND3_DIR
  HEAD_DIR      any head folder (the eye needs one to load; the heads are not used here)
  DATA_DIR      holds data/labels.csv + data/crops (streamsafe_data.tar)
  NEWCROPS_DIR  holds newcrops/labels_new.csv (+ crops)
  ROUNDk_DIR    holds the picks csv and <video>_unique.tar files of that round
LABELS.json: {"old": {"1": "0:RRUS...;1:...", "2": ..., "3": ...}, "new": {...}}
Writes X (fp16, unit norm), C_old, C_new ('-' = not R/M/S), G (game family; 'old' = training only), SRC.
env AUG=K adds K recoloured copies of every revealing (R) crop (SRC 'aug', same family, training only): the clothes
get new colours, skin and the dark background stay; an audit sheet OUT.aug.jpg shows a sample.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bench.train_heads import family
from streamsafe.config import Config
from streamsafe.models import build_models

SOURCES = ("old", "newcrops", "video", "video2", "clubs", "aug")
ROUNDS = {"1": ("picks.csv", "video"), "2": ("picks2.csv", "video2"), "3": ("picks3.csv", "clubs")}


def _ui_class(label: str) -> str:
    """label.html classes -> R/M/S (not_person counts as safe, as in every earlier head)."""
    if label in ("swim_underwear", "exposed", "revealing"):
        return "R"
    return "M" if label in ("shirtless_male", "revealing_male") else "S"


def recolor(rgb: np.ndarray, shift: int) -> np.ndarray:
    """Clothes in a new colour: saturated non-skin pixels (not red / orange) get their hue rotated by ``shift`` (OpenCV hue units,
    0-179); pale non-skin pixels (white / light grey clothes) are dyed that hue. Skin and the darkened background
    stay as they are."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = (ycc[..., 1] >= 133) & (ycc[..., 1] <= 173) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)
    h, sat, val = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    skin_hue = (h <= 25) | (h >= 165)  # red / orange: lit skin looks like this, so those pixels are never moved
    vivid = (sat >= 50) & (val >= 50) & ~skin & ~skin_hue
    pale = (sat < 50) & (val >= 110) & ~skin & ~skin_hue
    hsv[..., 0] = np.where(vivid, (h + shift) % 180, np.where(pale, shift % 180, h)).astype(np.uint8)
    hsv[..., 1] = np.where(pale, np.maximum(sat, 110), sat).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _codes(spec: str) -> dict[tuple[int, int], str]:
    out = {}
    for part in spec.split(";"):
        sheet, codes = part.split(":")
        out.update({(int(sheet), cell): c for cell, c in enumerate(codes)})
    return out


def main(argv: list[str]) -> None:
    out, labels = argv[0], json.loads(Path(argv[1]).read_text())
    data, newcrops, round_dirs = Path(argv[2]), Path(argv[3]), dict(zip(ROUNDS, map(Path, argv[4:7]), strict=True))
    images: list[np.ndarray] = []
    rows: list[tuple[str, str, str, str]] = []  # (C_old, C_new, G, SRC)

    def add(img: Image.Image, old: str, new: str, game: str, src: str) -> None:
        images.append(np.asarray(img.convert("RGB")))
        rows.append((old, new, game, src))

    for r in _rows(data / "data/labels.csv"):
        c = _ui_class(r["label"])
        add(Image.open(data / r["path"]), c, c, "old", "old")
    for r in _rows(newcrops / "newcrops/labels_new.csv"):
        c = _ui_class(r["label"])
        add(Image.open(newcrops / r["path"]), c, c, family(r["game"]), "newcrops")
    for k, (picks, src) in ROUNDS.items():
        old, new = _codes(labels["old"][k]), _codes(labels["new"][k])
        need: dict[str, dict[int, tuple[str, str]]] = {}
        for r in _rows(round_dirs[k] / picks):
            key = (int(r["sheet"]), int(r["cell"]))
            a, b = old.get(key, "U"), new.get(key, "U")
            if a in "RMS" or b in "RMS":
                need.setdefault(r["video"], {})[int(r["idx"])] = (a if a in "RMS" else "-", b if b in "RMS" else "-")
        for video, wanted in need.items():
            with tarfile.open(round_dirs[k] / f"{video}_unique.tar") as tf:
                for m in tf.getmembers():
                    j = int(Path(m.name).stem)
                    if j in wanted:
                        add(Image.open(io.BytesIO(tf.extractfile(m).read())), *wanted[j], family(video), src)
    k_aug = int(os.environ.get("AUG", 0))
    if k_aug:
        sample = []
        for i in [i for i, r in enumerate(rows) if r[1] == "R"]:
            for j in range(k_aug):
                images.append(recolor(images[i], 180 * (j + 1) // (k_aug + 1)))
                rows.append(("R", "R", rows[i][2], "aug"))
                if len(sample) < 48 and j == 0:
                    sample.append(np.hstack([cv2.resize(images[i], (64, 128)), cv2.resize(images[-1], (64, 128))]))
        sample += [np.zeros_like(sample[0])] * (-len(sample) % 8)
        sheet = np.vstack([np.hstack(sample[i : i + 8]) for i in range(0, len(sample), 8)])
        cv2.imwrite(f"{out}.aug.jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("LABELED", len(rows), {s: sum(1 for r in rows if r[3] == s) for s in SOURCES}, flush=True)
    eye = build_models(Config.from_env(scene_prior=False)).eye
    emb = np.concatenate([eye.embed(images[i : i + 64]) for i in range(0, len(images), 64)])
    c_old, c_new, g, src = (np.array(col) for col in zip(*rows, strict=True))
    np.savez_compressed(out, X=emb.astype(np.float16), C_old=c_old, C_new=c_new, G=g, SRC=src)
    for name, c in (("old", c_old), ("new", c_new)):
        print("COUNTS", name, {s: {k: int(((src == s) & (c == k)).sum()) for k in "RMS"} for s in sorted(set(src))}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
