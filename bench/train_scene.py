"""Train the explicit-scene head on labeled game frames (CPU, seconds).

Inputs: one or more scene_emb.npz files from bench.scene_probe (clip, t, emb) and a labels CSV
(npz_index,row,label) with label E = explicit scene (must be hidden whole-screen) or N = not.
Evaluation: leave-one-source-video-out (clip name without its start-second suffix), so a video never
helps judge itself.
The operating point favours precision (a whole-screen card on a normal scene is very visible): the
threshold where at most MAX_FP of the not-explicit frames would trigger, and the recall it gives.

usage: python -m bench.train_scene OUT_DIR LABELS.csv EMB0.npz [EMB1.npz ...]
Writes OUT_DIR/scene_h.txt (eye-head format) and OUT_DIR/scene_report.json; prints "SCENE_HEAD <json>".
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bench.train_heads import _quantise

MAX_FP = 0.01  # not-explicit frames allowed above the enter threshold
EXIT_FP = 0.05  # ... above the (lower) stay-on threshold


def source(clip: str) -> str:
    """gta5vu1_435 -> gta5vu1 (clips cut from one video share a source)."""
    return re.sub(r"_\d+$", "", clip)


def main(argv: list[str]) -> None:
    out, labels_csv, npz_paths = Path(argv[0]), argv[1], argv[2:]
    data = [np.load(p) for p in npz_paths]
    x, y, fam = [], [], []
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            d = data[int(r["npz_index"])]
            i = int(r["row"])
            x.append(d["emb"][i].astype(np.float32))
            y.append(r["label"].strip().upper() == "E")
            fam.append(source(str(d["clip"][i])))
    x, y, fam = np.stack(x), np.array(y), np.array(fam)
    p = np.full(len(y), np.nan)
    for g in sorted(set(fam)):
        tr = fam != g
        if y[tr].all() or not y[tr].any():
            continue
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(x[tr], y[tr])
        p[fam == g] = clf.predict_proba(x[fam == g])[:, 1]
    ok = ~np.isnan(p)
    neg = np.sort(p[ok & ~y])
    enter = float(neg[min(len(neg) - 1, int(np.ceil((1 - MAX_FP) * len(neg))))]) if len(neg) else 0.5
    stay = float(neg[min(len(neg) - 1, int(np.ceil((1 - EXIT_FP) * len(neg))))]) if len(neg) else 0.3
    rep = dict(
        n=len(y),
        explicit=int(y.sum()),
        families=sorted(set(fam)),
        lofo_auc=round(float(roc_auc_score(y[ok], p[ok])), 4) if len(set(y[ok])) == 2 else None,
        scene_enter=round(enter, 4),
        scene_exit=round(stay, 4),
        recall_at_enter=round(float((p[ok & y] >= enter).mean()), 4) if (ok & y).any() else None,
        recall_at_exit=round(float((p[ok & y] >= stay).mean()), 4) if (ok & y).any() else None,
        per_family={g: dict(n=int((fam == g).sum()), explicit=int(y[fam == g].sum()),
                            caught=round(float((p[(fam == g) & y] >= enter).mean()), 3) if y[fam == g].any() else None,
                            false=round(float((p[(fam == g) & ~y] >= enter).mean()), 3) if (~y[fam == g]).any() else None)
                    for g in sorted(set(fam))},
    )  # fmt: skip
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(x, y)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene_h.txt").write_text(_quantise(clf))
    (out / "scene_report.json").write_text(json.dumps(rep, indent=1))
    print("SCENE_HEAD " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
