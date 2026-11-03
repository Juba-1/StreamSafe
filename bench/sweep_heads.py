"""Compare head-training variants on one labeled set: which use of a hard-mining round helps?

Each variant is trained like bench/train_heads.py (same recipe, same LOFO threshold) and scored on
  - leave-one-game-family-out (LOFO) at 85% / 90% women recall: modest and men covered, women caught
  - the held-out mined test split: women / modest / men covered at the variant's own LOFO threshold,
    and AUC of revealing women vs everyone else (threshold-free ranking quality)

usage: python bench/sweep_heads.py LABELED_EMB.npz MINED_DIR LABELS.csv OLD_HEAD_DIR
Prints one "SWEEP <name> <json>" line per variant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from train_heads import load_mined, report, score_test

from streamsafe.models.heads import RiskHeads


def _lr(c: float) -> LogisticRegression:
    return LogisticRegression(max_iter=5000, C=c, class_weight="balanced")


def fit(x: np.ndarray, cls: np.ndarray, w: np.ndarray, c: float) -> tuple[LogisticRegression, LogisticRegression]:
    a, b = cls != "M", cls != "S"
    return _lr(c).fit(x[a], cls[a] == "R", sample_weight=w[a]), _lr(c).fit(x[b], cls[b] == "M", sample_weight=w[b])


def prisk(heads: tuple[LogisticRegression, LogisticRegression], x: np.ndarray) -> np.ndarray:
    return heads[0].predict_proba(x)[:, 1] * (1 - heads[1].predict_proba(x)[:, 1])


def lofo(x: np.ndarray, cls: np.ndarray, fam: np.ndarray, w: np.ndarray, c: float) -> np.ndarray:
    p = np.full(len(cls), np.nan)
    for g in sorted(set(fam) - {"old"}):
        test = fam == g
        p[test] = prisk(fit(x[~test], cls[~test], w[~test], c), x[test])
    return p


def _auc(p: np.ndarray, cls: np.ndarray) -> float:
    return round(float(roc_auc_score(cls == "R", p)), 4)


def main(argv: list[str]) -> None:
    base = np.load(argv[0])
    x0, c0, f0, s0 = base["X"].astype(np.float32), base["C"], base["G"], base["SRC"]
    mined = load_mined(Path(argv[1]), Path(argv[2]))
    tr, te = mined["train"], mined["test"]

    old = RiskHeads.load(argv[3])
    p_old = old.prisk(te["x"])
    out = score_test(p_old, old.thr, te["cls"], te["small"]) | {"auc": _auc(p_old, te["cls"])}
    print("SWEEP old_heads " + json.dumps(out), flush=True)

    # (name, which hard labels to add, weight of hard S crops, C)
    variants = [
        ("old_recipe_C2", "", 1.0, 2.0),
        ("all_hard", "RMS", 1.0, 2.0),
        ("hard_R_M_only", "RM", 1.0, 2.0),
        ("hard_S_w0.3", "RMS", 0.3, 2.0),
        ("hard_R_x5", "RMS", 1.0, 2.0),
        ("old_recipe_C0.5", "", 1.0, 0.5),
        ("all_hard_C0.5", "RMS", 1.0, 0.5),
    ]
    for name, use, w_s, c in variants:
        keep = np.isin(tr["cls"], list(use)) if use else np.zeros(len(tr["cls"]), bool)
        x = np.concatenate([x0, tr["x"][keep]])
        cls = np.concatenate([c0, tr["cls"][keep]])
        fam = np.concatenate([f0, tr["fam"][keep]])
        src = np.concatenate([s0, np.full(keep.sum(), "hard")])
        w = np.ones(len(cls))
        hard = src == "hard"
        w[hard & (cls == "S")] = w_s
        if name == "hard_R_x5":
            w[hard & (cls == "R")] = 5.0
        rep = report(lofo(x, cls, fam, w, c), cls, fam, src)
        thr85, thr90 = rep["at_0.85"]["thr"], rep["at_0.9"]["thr"]
        p = prisk(fit(x, cls, w, c), te["x"])
        res = dict(
            lofo85=rep["at_0.85"],
            lofo90=rep["at_0.9"],
            test85=score_test(p, thr85, te["cls"], te["small"]),
            test90=score_test(p, thr90, te["cls"], te["small"]),
            test_auc=_auc(p, te["cls"]),
        )
        print(f"SWEEP {name} " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
