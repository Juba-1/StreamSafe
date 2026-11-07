"""Find every labeled mined person again and keep what later steps need: the full-resolution eye embedding
(what the live engine sees), the analysis-frame embedding, the box and the person's COCO-17 keypoints.

Several mining rounds at once (each a MINED_DIR with meta.csv + emb.npy and Ali's labels R/M/S/N).
The mining sampler is re-run (same FPS / SECS); a person is the same when the recomputed analysis-frame
embedding matches the stored one (cosine >= 0.97) and the height agrees within 4 px.

usage: python -m bench.refind OUT.npz HEAD_DIR TAG=MINED_DIR=LABELS.csv [...] -- VIDEO [VIDEO ...]
env:   FPS=2 SECS=90 (must match the mining runs)
Writes OUT.npz: tag, id, split, video, label, h, box (n,4 analysis px), x_low, x_high (fp16), kp (n,17,2) in box
units (0..1, x from the box left, y from the box top), ks (n,17) keypoint scores. Prints "REFIND ..." lines.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.config import Config
from streamsafe.io.sources import VideoFileSource
from streamsafe.models import build_models
from streamsafe.types import ANALYSIS_SIZE

SAME_PERSON = 0.97
MAX_DH = 4


def wanted(specs: list[str]) -> dict[tuple[str, int], list[dict]]:
    """(video, sample index) -> labeled people to find, from every mining round."""
    want: dict[tuple[str, int], list[dict]] = {}
    for spec in specs:
        tag, mined, labels_csv = spec.split("=")
        with open(labels_csv) as f:
            labels = {int(r["id"]): r["label"].strip().upper() for r in csv.DictReader(f)}
        emb = np.load(Path(mined) / "emb.npy").astype(np.float32)
        with open(Path(mined) / "meta.csv") as f:
            for r in csv.DictReader(f):
                i = int(r["id"])
                if labels.get(i, "U") in ("R", "M", "S", "N"):
                    item = dict(tag=tag, id=i, split=r["split"], video=r["video"], h=int(r["h"]), label=labels[i], emb=emb[i])
                    want.setdefault((r["video"], int(r["frame"])), []).append(item)
    return want


def box_units(kp: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return (kp - np.array([x1, y1], np.float32)) / np.array([max(1, x2 - x1), max(1, y2 - y1)], np.float32)


def main(argv: list[str]) -> None:
    cut = argv.index("--")
    out, head_dir, specs, videos = argv[0], argv[1], argv[2:cut], argv[cut + 1 :]
    fps, secs = float(os.environ.get("FPS", 2)), float(os.environ.get("SECS", 90))
    want = wanted(specs)
    cfg = Config.from_env(head_dir=head_dir, scene_prior=False)
    models = build_models(cfg)
    analyzer = Analyzer(cfg, models, output_scale=1.0)
    eye = models.eye
    rows: list[dict] = []
    for path in videos:
        name = Path(path).stem
        if not any(v == name for v, _ in want):
            continue
        source = VideoFileSource(path, fps, secs)
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
            for item in todo:
                j = int(np.argmax(e_low @ item["emb"]))
                if float(e_low[j] @ item["emb"]) < SAME_PERSON or abs(people[j].height - item["h"]) > MAX_DH:
                    continue
                kp, ks = models.pose(rgb, [people[j].box])[0]
                found += 1
                rows.append(
                    {k2: item[k2] for k2 in ("tag", "id", "split", "video", "label", "h")}
                    | dict(box=people[j].box, x_low=e_low[j], x_high=e_high[j], kp=box_units(kp, people[j].box), ks=ks)
                )
        source.close()
        print("REFIND", name, "found", found, "of", sum(len(v) for (n, _), v in want.items() if n == name), flush=True)
    cols = {k: np.array([r[k] for r in rows]) for k in ("tag", "id", "split", "video", "label", "h", "box", "kp", "ks")}
    np.savez_compressed(
        out,
        x_low=np.stack([r["x_low"] for r in rows]).astype(np.float16),
        x_high=np.stack([r["x_high"] for r in rows]).astype(np.float16),
        **cols,
    )
    print("REFIND_TOTAL", len(rows), "of", sum(len(v) for v in want.values()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
