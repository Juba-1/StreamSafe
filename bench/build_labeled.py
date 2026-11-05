"""Embed every labeled game crop of the earlier rounds with the engine's own eye, keeping two labelings side by side.

Rounds: ``old`` + ``newcrops`` (label.html, labeled by Ali), video rounds 1-2 and clubs round 3 (contact sheets, labeled by
Claude; ``LABELS.json`` holds the original codes and the codes after Ali's review, per round and sheet). A crop is kept
when either labeling makes it R/M/S, so recipes trained on the old and on the corrected labels see the same rows.

usage: python bench/build_labeled.py OUT.npz LABELS.json DATA_DIR NEWCROPS_DIR ROUND1_DIR ROUND2_DIR ROUND3_DIR
  DATA_DIR      holds data/labels.csv + data/crops (streamsafe_data.tar)
  NEWCROPS_DIR  holds newcrops/labels_new.csv (+ crops)
  ROUNDk_DIR    holds the picks csv and <video>_unique.tar files of that round
LABELS.json: {"old": {"1": "0:RRUS...;1:...", "2": ..., "3": ...}, "new": {...}}
Writes X (fp16, unit norm), C_old, C_new ('-' = not R/M/S), G (game family; 'old' = training only), SRC.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from bench.train_heads import family
from streamsafe.config import Config
from streamsafe.models import build_models

ROUNDS = {"1": ("picks.csv", "video"), "2": ("picks2.csv", "video2"), "3": ("picks3.csv", "clubs")}


def _ui_class(label: str) -> str:
    """label.html classes -> R/M/S (not_person counts as safe, as in every earlier head)."""
    if label in ("swim_underwear", "exposed", "revealing"):
        return "R"
    return "M" if label in ("shirtless_male", "revealing_male") else "S"


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
    print("LABELED", len(rows), {s: sum(1 for r in rows if r[3] == s) for s in ("old", "newcrops", "video", "video2", "clubs")}, flush=True)
    eye = build_models(Config.from_env(scene_prior=False)).eye
    emb = np.concatenate([eye.embed(images[i : i + 64]) for i in range(0, len(images), 64)])
    c_old, c_new, g, src = (np.array(col) for col in zip(*rows, strict=True))
    np.savez_compressed(out, X=emb.astype(np.float16), C_old=c_old, C_new=c_new, G=g, SRC=src)
    for name, c in (("old", c_old), ("new", c_new)):
        print("COUNTS", name, {s: {k: int(((src == s) & (c == k)).sum()) for k in "RMS"} for s in sorted(set(src))}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
