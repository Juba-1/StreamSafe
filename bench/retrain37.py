"""Run 37: retrain the risk heads after Ali's review of the old rounds, with and without real photos.

Recipes (all: gated heads, logistic regression C=2 balanced, threshold = 85% women recall in leave-one-game-family-out):
    old_labels       the earlier rounds as Claude labeled them (+ Ali's hard round)       = today's recipe
    ali_labels       the same crops after Ali's review
    ali+photos       + Open Images photos (training only, never in a test fold)
    ali+photos_half  photos at half weight
Every recipe is SCORED with Ali's labels, so the numbers compare directly. Photos never enter a test fold.

usage: python bench/retrain37.py OUT_DIR LABELED.npz PHOTOS.npz MINED_DIR LABELS.csv OLD_HEAD_DIR
Writes OUT_DIR/<recipe>/{fast_h6,male_h6,thr_h6}.txt and OUT_DIR/report37.json; prints "R37 <json>".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bench.train_heads import FAST_FILE, MALE_FILE, THR_FILE, _quantise, load_mined, report, score_test
from streamsafe.models.heads import RiskHeads


def fit(x: np.ndarray, cls: np.ndarray, w: np.ndarray) -> tuple[LogisticRegression, LogisticRegression]:
    def lr() -> LogisticRegression:
        return LogisticRegression(max_iter=5000, C=2.0, class_weight="balanced")

    a, b = cls != "M", cls != "S"
    return lr().fit(x[a], cls[a] == "R", sample_weight=w[a]), lr().fit(x[b], cls[b] == "M", sample_weight=w[b])


def prisk(heads: tuple[LogisticRegression, LogisticRegression], x: np.ndarray) -> np.ndarray:
    return heads[0].predict_proba(x)[:, 1] * (1 - heads[1].predict_proba(x)[:, 1])


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    lab, photos = np.load(argv[1]), np.load(argv[2])
    mined = load_mined(Path(argv[3]), Path(argv[4]))
    tr, te = mined["train"], mined["test"]
    x = np.concatenate([lab["X"].astype(np.float32), tr["x"]])
    fam = np.concatenate([lab["G"], tr["fam"]])
    src = np.concatenate([lab["SRC"], np.full(len(tr["cls"]), "hard")])
    c_old = np.concatenate([lab["C_old"], tr["cls"]])
    c_new = np.concatenate([lab["C_new"], tr["cls"]])
    px, pc = photos["X"].astype(np.float32), photos["C"]
    recipes = {
        "old_labels": (c_old, 0.0),
        "ali_labels": (c_new, 0.0),
        "ali+photos": (c_new, 1.0),
        "ali+photos_half": (c_new, 0.5),
    }
    scored = c_new != "-"  # every recipe is judged on Ali's labels
    rep: dict = {
        "counts_old": {c: int((c_old == c).sum()) for c in "RMS"},
        "counts_new": {c: int((c_new == c).sum()) for c in "RMS"},
        "photos": {c: int((pc == c).sum()) for c in "RMS"},
    }
    old_heads = RiskHeads.load(argv[5])
    rep["test_old_heads"] = score_test(old_heads.prisk(te["x"]), old_heads.thr, te["cls"], te["small"])
    for name, (cls, pw) in recipes.items():
        use = cls != "-"
        p = np.full(len(cls), np.nan)
        for g in sorted(set(fam[scored]) - {"old"}):
            test = (fam == g) & scored
            train = (fam != g) & use
            xs, cs, ws = x[train], cls[train], np.ones(int(train.sum()))
            if pw > 0:
                xs, cs, ws = np.concatenate([xs, px]), np.concatenate([cs, pc]), np.concatenate([ws, np.full(len(pc), pw)])
            p[test] = prisk(fit(xs, cs, ws), x[test])
        known = scored & (fam != "old")
        r = report(np.where(known, p, np.nan), np.where(scored, c_new, "-"), fam, src)
        r["lofo_auc_women_vs_rest"] = round(float(roc_auc_score(c_new[known] == "R", p[known])), 4)
        xs, cs, ws = x[use], cls[use], np.ones(int(use.sum()))
        if pw > 0:
            xs, cs, ws = np.concatenate([xs, px]), np.concatenate([cs, pc]), np.concatenate([ws, np.full(len(pc), pw)])
        heads = fit(xs, cs, ws)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        thr = r["at_0.85"]["thr"]
        for fname, text in ((FAST_FILE, _quantise(heads[0])), (MALE_FILE, _quantise(heads[1])), (THR_FILE, f"{thr:.4f}")):
            (d / fname).write_text(text)
        new = RiskHeads.load(d)
        r["test"] = score_test(new.prisk(te["x"]), new.thr, te["cls"], te["small"])
        if pw > 0:  # how the photo-trained heads judge the photos themselves (sanity, not a test)
            pp = new.prisk(px) >= new.thr
            r["photos_covered"] = {c: round(float(pp[pc == c].mean()), 4) for c in "RMS" if (pc == c).any()}
        rep[name] = r
        print("R37", name, json.dumps(r), flush=True)
    (out / "report37.json").write_text(json.dumps(rep, indent=1))
    print("R37_REPORT " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
