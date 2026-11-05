"""Real photos for the eye: people in swimwear / underwear and people in clearly modest clothes, from Open Images.

Open Images V7 (annotations CC BY 4.0, images listed as CC BY 2.0 - DOC-CLAIM) has boxes for Woman / Man and for
clothing. A person box is labeled from the clothing boxes inside it (groups, each with its own cap):
    R   woman + Swimwear or Brassiere
    M   man + Swimwear (no Brassiere in the photo)
    SW  woman + a long Dress (down to the feet) or Suit / Coat / Jacket / Jeans / Trousers   -> class S
    SM  man + Suit / Coat / Jacket / Jeans / Trousers                                       -> class S
    (S only when no swimwear / underwear / miniskirt / shorts is on that person)
A person overlapping someone of the other sex (IoU > 0.3) is skipped: the clothing box could be theirs.
Images with a Girl or Boy box are skipped (no children), as are drawings (IsDepiction) and group boxes (IsGroupOf).
Each photo goes through the engine's own detector and eye crop (full-resolution path), so the eye sees what it sees live.
Photos are never written out; only embeddings and a small audit sheet per class.

usage: python bench/oi_photos.py OUT_DIR
env:   HEAD_DIR=<any head dir, the eye needs one to load>  N_R=1500 N_M=600 N_SW=1500 N_SM=500 WORKERS=32
Writes OUT_DIR/photos_emb.npz (X fp16, C, G='old' = training only, SRC='photo', TAG = group:rule), OUT_DIR/sheet_<group>.jpg.
"""

from __future__ import annotations

import csv
import io
import os
import random
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.config import Config
from streamsafe.models import build_models
from streamsafe.types import ANALYSIS_SIZE

CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
BOXES_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"
PEOPLE = ("Woman", "Man")
CHILDREN = ("Girl", "Boy")
REVEALING = ("Swimwear", "Brassiere")
AMBIGUOUS = ("Miniskirt", "Shorts")
MODEST = ("Suit", "Coat", "Jacket", "Jeans", "Trousers")
DRESS = "Dress"
LONG_DRESS_BOTTOM = 0.12  # the dress ends within this share of the person height from their feet
LONG_DRESS_SPAN = 0.55  # and covers at least this share of their height
OTHER_SEX_IOU = 0.3
GROUP_CLASS = {"R": "R", "M": "M", "SW": "S", "SM": "S"}
INSIDE = 0.8  # share of a clothing box that must lie inside the person box
MIN_BOX_H = 0.25  # person box height, share of the image height
MATCH_IOU = 0.5
SAMPLE_IMAGES = 15000  # per group (revealing / modest) before matching
CHUNK = 256  # photos in memory at once


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def _inside(c: np.ndarray, p: np.ndarray) -> bool:
    w = max(0.0, min(c[1], p[1]) - max(c[0], p[0]))
    h = max(0.0, min(c[3], p[3]) - max(c[2], p[2]))
    area = (c[1] - c[0]) * (c[3] - c[2])
    return area > 0 and w * h / area >= INSIDE


def _box(row: pd.Series) -> np.ndarray:
    return row[["XMin", "XMax", "YMin", "YMax"]].to_numpy(float)


def _iou_frac(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - w * h
    return w * h / union if union > 0 else 0.0


def _long_dress(d: np.ndarray, p: np.ndarray) -> bool:
    ph = p[3] - p[2]
    return d[3] >= p[3] - LONG_DRESS_BOTTOM * ph and d[3] - d[2] >= LONG_DRESS_SPAN * ph


def _label(woman: bool, worn: dict[str, list[np.ndarray]], pb: np.ndarray, bra_in_photo: bool) -> tuple[str, str] | None:
    """(group, rule) for one person, or None when the clothes do not decide it cleanly."""
    revealing = sorted(set(worn) & set(REVEALING))
    if revealing:
        if woman:
            return "R", "+".join(revealing)
        return ("M", "Swimwear") if "Swimwear" in worn and not bra_in_photo else None
    if set(worn) & set(AMBIGUOUS):
        return None
    modest = sorted(set(worn) & set(MODEST))
    if woman and any(_long_dress(d, pb) for d in worn.get(DRESS, [])):
        modest.append("LongDress")
    if not modest:
        return None
    return ("SW" if woman else "SM"), "+".join(modest)


def candidates(work: Path) -> list[tuple[str, str, str, tuple[float, float, float, float]]]:
    """(image id, group R/M/SW/SM, rule tag, person box as XMin,XMax,YMin,YMax fractions)."""
    names = pd.read_csv(_download(CLASSES_URL, work / "classes.csv"), header=None, names=["mid", "name"])
    mid = {n: m for m, n in zip(names["mid"], names["name"], strict=True)}
    wanted = {mid[n]: n for n in (*PEOPLE, *CHILDREN, *REVEALING, *AMBIGUOUS, *MODEST, DRESS) if n in mid}
    boxes = []
    cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"]
    for chunk in pd.read_csv(_download(BOXES_URL, work / "boxes.csv"), usecols=cols, chunksize=2_000_000):
        chunk = chunk[chunk["LabelName"].isin(wanted)]
        boxes.append(chunk.assign(name=chunk["LabelName"].map(wanted)))
    df = pd.concat(boxes)
    kids = set(df.loc[df["name"].isin(CHILDREN), "ImageID"])
    drawn = set(df.loc[df["IsDepiction"] == 1, "ImageID"])
    df = df[~df["ImageID"].isin(kids | drawn) & (df["IsGroupOf"] == 0)]
    rng = random.Random(0)  # only a sample of images is worth the slow per-person matching below
    sample = set()
    for names_ in (REVEALING, MODEST, (DRESS,)):
        ids = sorted(set(df.loc[df["name"].isin(names_), "ImageID"]))
        rng.shuffle(ids)
        sample.update(ids[:SAMPLE_IMAGES])
    df = df[df["ImageID"].isin(sample)]
    out = []
    for image_id, g in df.groupby("ImageID"):
        people = g[g["name"].isin(PEOPLE)]
        if people.empty:
            continue
        clothes = [(c["name"], _box(c)) for _, c in g[~g["name"].isin(PEOPLE)].iterrows()]
        bra_in_photo = any(n == "Brassiere" for n, _ in clothes)
        sexed = [(p["name"] == "Woman", _box(p)) for _, p in people.iterrows()]
        for woman, pb in sexed:
            if pb[3] - pb[2] < MIN_BOX_H:
                continue
            if any(w != woman and _iou_frac(pb, ob) > OTHER_SEX_IOU for w, ob in sexed):
                continue
            worn: dict[str, list[np.ndarray]] = defaultdict(list)
            for n, cb in clothes:
                if _inside(cb, pb):
                    worn[n].append(cb)
            label = _label(woman, worn, pb, bra_in_photo)
            if label is not None:
                out.append((image_id, label[0], label[1], tuple(pb)))
    return out


def _fetch(image_id: str) -> np.ndarray | None:
    try:
        with urllib.request.urlopen(IMAGE_URL.format(image_id), timeout=20) as r:
            return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
    except Exception:
        return None


def _frames(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    """The photo letterboxed into an analysis frame and into a 2x 'native' frame (as a 1080p stream)."""
    aw, ah = ANALYSIS_SIZE
    h, w = rgb.shape[:2]
    s = min(aw / w, ah / h)
    nw, nh = int(w * s), int(h * s)
    ox, oy = (aw - nw) // 2, (ah - nh) // 2
    small = np.zeros((ah, aw, 3), np.uint8)
    small[oy : oy + nh, ox : ox + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    full = np.zeros((2 * ah, 2 * aw, 3), np.uint8)
    full[2 * oy : 2 * oy + 2 * nh, 2 * ox : 2 * ox + 2 * nw] = cv2.resize(rgb, (2 * nw, 2 * nh), interpolation=cv2.INTER_AREA)
    return small, full, s, (ox, oy)


def _iou(a: tuple, b: tuple) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    defaults = {"R": 1500, "M": 600, "SW": 1500, "SM": 500}
    caps = {k: int(os.environ.get(f"N_{k}", v)) for k, v in defaults.items()}
    cands = candidates(out)
    by: dict[str, list] = defaultdict(list)
    for c in cands:
        by[c[1]].append(c)
    print("OI_CANDIDATES", {k: len(v) for k, v in by.items()}, flush=True)
    rng = random.Random(0)
    pick = []
    for k, v in by.items():
        rng.shuffle(v)
        seen: set[str] = set()
        for c in v:  # one person per photo: more distinct photos
            if c[0] not in seen and len(seen) < caps[k]:
                seen.add(c[0])
                pick.append(c)
    cfg = Config.from_env(scene_prior=False)
    analyzer = Analyzer(cfg, build_models(cfg), output_scale=1.0)
    xs, rows, sheet = [], [], defaultdict(list)
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 32))) as pool:
        for at in range(0, len(pick), CHUNK):
            part = pick[at : at + CHUNK]
            for item, rgb in zip(part, pool.map(_fetch, [p[0] for p in part]), strict=True):
                _add(analyzer, item, rgb, xs, rows, sheet)
        print("OI_EMBEDDED", len(rows), flush=True)
    _save(out, xs, rows, sheet, len(pick))


def _add(analyzer: Analyzer, item: tuple, rgb: np.ndarray | None, xs: list, rows: list, sheet: dict) -> None:
    """Find the annotated person among the engine's detections and embed the engine's own eye crop."""
    if rgb is None:
        return
    image_id, group, tag, pb = item
    small, full, s, (ox, oy) = _frames(rgb)
    h, w = rgb.shape[:2]
    want = (ox + pb[0] * w * s, oy + pb[2] * h * s, ox + pb[1] * w * s, oy + pb[3] * h * s)
    people, crops = analyzer.detect(small, full)
    if not people:
        return
    j = int(np.argmax([_iou(want, p.box) for p in people]))
    if _iou(want, people[j].box) < MATCH_IOU:
        return
    xs.append(analyzer.models.eye.embed([crops[j]])[0])
    rows.append((GROUP_CLASS[group], f"{group}:{tag}", image_id))
    if len(sheet[group]) < 48:
        sheet[group].append(cv2.resize(crops[j], (96, 192), interpolation=cv2.INTER_AREA))


def _save(out: Path, xs: list, rows: list, sheet: dict, n_picked: int) -> None:
    c = np.array([r[0] for r in rows])
    np.savez_compressed(
        out / "photos_emb.npz",
        X=np.array(xs, np.float16),
        C=c,
        G=np.full(len(c), "old"),
        SRC=np.full(len(c), "photo"),
        TAG=np.array([r[1] for r in rows]),
    )
    with open(out / "photos.csv", "w", newline="") as f:
        csv.writer(f).writerows([("class", "tag", "image_id"), *rows])  # attribution: Open Images train image ids
    for k, cells in sheet.items():
        cells += [np.zeros_like(cells[0])] * (-len(cells) % 12)
        grid = np.vstack([np.hstack(cells[i : i + 12]) for i in range(0, len(cells), 12)])
        cv2.imwrite(str(out / f"sheet_{k}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for name in ("boxes.csv", "classes.csv"):
        (out / name).unlink(missing_ok=True)
    print("OI_DONE", {k: int((c == k).sum()) for k in "RMS"}, "of", n_picked, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
