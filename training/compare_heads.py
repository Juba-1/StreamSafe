"""Compare head-training recipes on the same labeled crops and write the heads of each.

Recipes (all: the gated-head recipe of training/common.py, threshold = 85% women recall in leave-one-game-family-out),
chosen with RECIPES=name,name (default: all):
    old_labels    the earlier rounds as Claude labeled them (+ Ali's hard round)
    ali           the same crops after Ali's review
    ali_aug       + recoloured copies of the revealing crops (SRC 'aug', from build_labeled AUG=K)
    ali_photos    + Open Images photos at half weight
    ali_aug_photos  both
Every recipe is SCORED with Ali's labels on real (not recoloured) game crops, so the numbers compare directly.
env EXTRA=a.npz[,b.npz] adds examples from training/add_examples.py (e.g. a modest character the eye covered) to
the TRAINING of every recipe, never to a test fold; EXTRA_WEIGHT (default 1) is their sample weight.
Recoloured copies of a game family are left out of training while that family is tested; photos never enter a test fold.

usage: python -m training.compare_heads OUT_DIR LABELED.npz PHOTOS.npz MINED_DIR LABELS.csv OLD_HEAD_DIR
Writes OUT_DIR/<recipe>/{fast_h6,male_h6,thr_h6}.txt and OUT_DIR/compare.json; prints "R37 <json>".
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from streamsafe.models.heads import RiskHeads
from training.common import fit, load_mined, prisk, report, score_test, write_heads


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
    weight = np.ones(len(x))
    for path in filter(None, os.environ.get("EXTRA", "").split(",")):
        ex = np.load(path)
        n = len(ex["C"])
        x = np.concatenate([x, ex["X"].astype(np.float32)])
        fam = np.concatenate([fam, ex["G"]])
        src = np.concatenate([src, np.full(n, "extra")])
        c_old, c_new = np.concatenate([c_old, ex["C"]]), np.concatenate([c_new, ex["C"]])
        weight = np.concatenate([weight, np.full(n, float(os.environ.get("EXTRA_WEIGHT", 1.0)))])
    px, pc = photos["X"].astype(np.float32), photos["C"]
    all_recipes = {  # name: (labels, use recoloured copies, photo weight)
        "old_labels": (c_old, False, 0.0),
        "ali": (c_new, False, 0.0),
        "ali_aug": (c_new, True, 0.0),
        "ali_photos": (c_new, False, 0.5),
        "ali_aug_photos": (c_new, True, 0.5),
    }
    wanted = os.environ.get("RECIPES")
    recipes = {k: v for k, v in all_recipes.items() if not wanted or k in wanted.split(",")}
    aug = src == "aug"
    scored = (c_new != "-") & ~aug & (src != "extra")  # every recipe is judged on Ali's labels, real crops only
    rep: dict = {
        "counts_old": {c: int((c_old == c).sum()) for c in "RMS"},
        "counts_new": {c: int((c_new == c).sum()) for c in "RMS"},
        "photos": {c: int((pc == c).sum()) for c in "RMS"},
    }
    old_heads = RiskHeads.load(argv[5])
    rep["test_old_heads"] = score_test(old_heads.prisk(te["x"]), old_heads.thr, te["cls"], te["small"])
    for name, (cls, with_aug, pw) in recipes.items():
        use = (cls != "-") & (with_aug | ~aug)
        p = np.full(len(cls), np.nan)
        for g in sorted(set(fam[scored]) - {"old"}):
            test = (fam == g) & scored
            train = (fam != g) & use
            xs, cs, ws = x[train], cls[train], weight[train]
            if pw > 0:
                xs, cs, ws = np.concatenate([xs, px]), np.concatenate([cs, pc]), np.concatenate([ws, np.full(len(pc), pw)])
            p[test] = prisk(fit(xs, cs, ws), x[test])
        known = scored & (fam != "old")
        r = report(np.where(known, p, np.nan), np.where(scored, c_new, "-"), fam, src)
        r["lofo_auc_women_vs_rest"] = round(float(roc_auc_score(c_new[known] == "R", p[known])), 4)
        xs, cs, ws = x[use], cls[use], weight[use]
        if pw > 0:
            xs, cs, ws = np.concatenate([xs, px]), np.concatenate([cs, pc]), np.concatenate([ws, np.full(len(pc), pw)])
        d = out / name
        write_heads(d, fit(xs, cs, ws), r["at_0.85"]["thr"])
        new = RiskHeads.load(d)
        r["test"] = score_test(new.prisk(te["x"]), new.thr, te["cls"], te["small"])
        if pw > 0:  # how the photo-trained heads judge the photos themselves (sanity, not a test)
            pp = new.prisk(px) >= new.thr
            r["photos_covered"] = {c: round(float(pp[pc == c].mean()), 4) for c in "RMS" if (pc == c).any()}
        rep[name] = r
        print("HEADS_CMP", name, json.dumps(r), flush=True)
    (out / "compare.json").write_text(json.dumps(rep, indent=1))
    print("HEADS_CMP_REPORT " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
