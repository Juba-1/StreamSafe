"""Does the eye judge small people better when its crop comes from the full-resolution frame?

Re-runs the mining sampler (same FPS / SECS) over the given videos, finds every labeled mined crop again
(same video, same sample index, closest height, embedding check) and scores it twice with the same heads:
the crop cut from the 960x540 analysis frame (as today) and the crop cut from the native frame.

usage: python bench/hires_eye.py OUT_DIR HEAD_DIR MINED_DIR LABELS.csv VIDEO [VIDEO ...]
env:   FPS=2 SECS=90 (must match the mining run)
Prints "HIRES_VIDEO ..." per video and "HIRES_REPORT <json>"; writes OUT_DIR/hires.csv and a contact
sheet OUT_DIR/pairs.jpg (low-res crop | high-res crop) of the revealing women.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.config import Config
from streamsafe.io.sources import VideoFileSource
from streamsafe.models import build_models
from streamsafe.types import ANALYSIS_SIZE

SAME_PERSON = 0.97  # cosine between the stored and the recomputed analysis-frame embedding
MAX_DH = 4  # analysis px


def _labels(mined: Path, labels_csv: Path) -> dict[tuple[str, int], list[dict]]:
    with open(labels_csv) as f:
        labels = {int(r["id"]): r["label"].strip().upper() for r in csv.DictReader(f)}
    emb = np.load(mined / "emb.npy").astype(np.float32)
    want: dict[tuple[str, int], list[dict]] = {}
    with open(mined / "meta.csv") as f:
        for r in csv.DictReader(f):
            lab = labels.get(int(r["id"]), "U")
            if lab in ("R", "M", "S"):
                key = (r["video"], int(r["frame"]))
                want.setdefault(key, []).append(dict(id=int(r["id"]), split=r["split"], h=int(r["h"]), label=lab, emb=emb[int(r["id"])]))
    return want


def _score(p: np.ndarray, thr: float, cls: np.ndarray) -> dict:
    hit = p >= thr
    groups = (("women", "R"), ("modest", "S"), ("men", "M"))
    out = {f"{name}_covered": round(float(hit[cls == c].mean()), 4) for name, c in groups if (cls == c).any()}
    if len(set(cls == "R")) == 2:
        out["auc_women_vs_rest"] = round(float(roc_auc_score(cls == "R", p)), 4)
    return out


def main(argv: list[str]) -> None:
    out, head_dir, mined, labels_csv, videos = Path(argv[0]), argv[1], Path(argv[2]), Path(argv[3]), argv[4:]
    out.mkdir(parents=True, exist_ok=True)
    fps, secs = float(os.environ.get("FPS", 2)), float(os.environ.get("SECS", 90))
    want = _labels(mined, labels_csv)
    cfg = Config.from_env(head_dir=head_dir, scene_prior=False)
    analyzer = Analyzer(cfg, build_models(cfg), output_scale=1.0)
    eye, thr = analyzer.models.eye, analyzer.thr
    rows, pairs = [], []
    for path in videos:
        name = Path(path).stem
        source = VideoFileSource(path, fps, secs)  # native size (16:9)
        found = 0
        for k, frame in enumerate(source.frames()):
            todo = want.get((name, k))
            if not todo:
                continue
            full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.cvtColor(cv2.resize(frame, ANALYSIS_SIZE), cv2.COLOR_BGR2RGB)  # exactly as the mining run
            people, low = analyzer.detect(rgb)
            if not people:
                continue
            _, high = analyzer.detect(rgb, full)
            e_low, e_high = eye.embed(low), eye.embed(high)
            p_low, p_high = eye.prisk(e_low), eye.prisk(e_high)
            for item in todo:
                j = int(np.argmax(e_low @ item["emb"]))
                if float(e_low[j] @ item["emb"]) < SAME_PERSON or abs(people[j].height - item["h"]) > MAX_DH:
                    continue
                found += 1
                rows.append(
                    dict(id=item["id"], split=item["split"], video=name, h=item["h"], label=item["label"])
                    | dict(p_low=float(p_low[j]), p_high=float(p_high[j]))
                )
                if item["label"] == "R":
                    pairs.append((low[j], high[j]))
        source.close()
        labeled = sum(len(v) for (n, _), v in want.items() if n == name)
        print("HIRES_VIDEO", name, "native", source.size, "matched", found, "of", labeled, flush=True)
    with open(out / "hires.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rep: dict = {"thr": round(thr, 4)}
    for split in ("test", "train", "all"):
        sel = [r for r in rows if split == "all" or r["split"] == split]
        if not sel:
            continue
        cls = np.array([r["label"] for r in sel])
        rep[split] = dict(
            n={c: int((cls == c).sum()) for c in "RSM"},
            low=_score(np.array([r["p_low"] for r in sel]), thr, cls),
            high=_score(np.array([r["p_high"] for r in sel]), thr, cls),
        )
    print("HIRES_REPORT " + json.dumps(rep), flush=True)
    if pairs:
        cells = [np.hstack([_cell(a), _cell(b)]) for a, b in pairs[:40]]
        cells += [np.zeros_like(cells[0])] * (-len(cells) % 8)
        sheet = np.vstack([np.hstack(cells[i : i + 8]) for i in range(0, len(cells), 8)])
        cv2.imwrite(str(out / "pairs.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _cell(crop: np.ndarray, w: int = 96, h: int = 192) -> np.ndarray:
    """What the eye sees, shrunk for the sheet: the crop stretched to the eye's aspect."""
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main(sys.argv[1:])
