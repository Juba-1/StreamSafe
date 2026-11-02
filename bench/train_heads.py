"""Train the gated risk heads (the recipe of runs 16-21) on every labeled crop, optionally plus a
newly labeled hard-mining round, and report how they generalise to games they never saw.

    p(cover) = women(x) * (1 - male(x))     women: R vs S (men left out), male: M vs R
    logistic regression, C=2, balanced classes; threshold = 85% women recall in leave-one-game-family-out

usage: python bench/train_heads.py OUT_DIR LABELED_EMB.npz [MINED_DIR LABELS.csv [OLD_HEAD_DIR]]
LABELS.csv: id,label with label R (revealing woman), M (revealing man), S (safe), N (not a person), U (unsure);
only R/M/S are used. The mined "test" split (held-out videos) is never trained on; it is scored with the
new heads and, when OLD_HEAD_DIR is given, with the old ones.
Writes OUT_DIR/{fast_h6,male_h6,thr_h6}.txt (the engine's head files) and OUT_DIR/report.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from streamsafe.models.heads import FAST_FILE, MALE_FILE, THR_FILE, RiskHeads

RECALLS = (0.85, 0.90)
FAMILIES = {  # clip-name prefix -> game family (same rule as the earlier rounds); anything else is GTA
    "cyber": "cyber", "yakuza": "yakuza", "sr22": "saintsrow", "sr3": "saintsrow", "sims4": "sims4", "tlou2": "tlou2", "wd2": "watchdogs",
    "wdl": "watchdogs", "sdogs": "sleepingdogs", "di2": "deadisland", "fc": "farcry", "hitman": "hitman", "rdr2": "rdr2",
    "lizzie": "lizzie",
}  # fmt: skip


def family(video: str) -> str:
    return next((name for prefix, name in FAMILIES.items() if video.startswith(prefix)), "gta")


def _lr() -> LogisticRegression:
    return LogisticRegression(max_iter=5000, C=2.0, class_weight="balanced")


def fit(x: np.ndarray, cls: np.ndarray) -> tuple[LogisticRegression, LogisticRegression]:
    women = _lr().fit(x[cls != "M"], cls[cls != "M"] == "R")
    male = _lr().fit(x[cls != "S"], cls[cls != "S"] == "M")
    return women, male


def prisk(heads: tuple[LogisticRegression, LogisticRegression], x: np.ndarray) -> np.ndarray:
    women, male = heads
    return women.predict_proba(x)[:, 1] * (1 - male.predict_proba(x)[:, 1])


def lofo(x: np.ndarray, cls: np.ndarray, fam: np.ndarray) -> np.ndarray:
    """Out-of-family p for every crop of a known family ('old' crops are training-only)."""
    p = np.full(len(cls), np.nan)
    for g in sorted(set(fam) - {"old"}):
        test = fam == g
        p[test] = prisk(fit(x[~test], cls[~test]), x[test])
    return p


def _share(mask: np.ndarray, hit: np.ndarray) -> float | None:
    return round(float(hit[mask].mean()), 4) if mask.any() else None


def report(p: np.ndarray, cls: np.ndarray, fam: np.ndarray, src: np.ndarray) -> dict:
    known = fam != "old"
    women = known & (cls == "R")
    out = {}
    for rec in RECALLS:
        thr = float(np.quantile(p[women], 1 - rec))
        hit = p >= thr
        out[f"at_{rec}"] = dict(
            thr=round(thr, 4),
            modest_covered=_share(known & (cls == "S"), hit),
            men_covered=_share(known & (cls == "M"), hit),
            video_women_caught=_share(women & np.isin(src, ["video", "video2"]), hit),
            hard_women_caught=_share(women & (src == "hard"), hit),
            hard_modest_covered=_share(known & (cls == "S") & (src == "hard"), hit),
        )
    return out


def score_test(p: np.ndarray, thr: float, cls: np.ndarray, small: np.ndarray) -> dict:
    hit = p >= thr
    return dict(
        thr=round(thr, 4),
        women_caught=_share(cls == "R", hit),
        small_mid_women_caught=_share((cls == "R") & small, hit),
        modest_covered=_share(cls == "S", hit),
        men_covered=_share(cls == "M", hit),
        n=dict(R=int((cls == "R").sum()), S=int((cls == "S").sum()), M=int((cls == "M").sum())),
    )


def _quantise(head: LogisticRegression) -> str:
    w = head.coef_.ravel()
    scale = 32000 / np.abs(w).max()
    return f"{scale};{head.intercept_[0]:.5f};" + ",".join(str(round(v)) for v in w * scale)


def load_mined(mined: Path, labels_csv: Path) -> dict[str, dict]:
    with open(labels_csv) as f:
        labels = {int(r["id"]): r["label"].strip().upper() for r in csv.DictReader(f)}
    with open(mined / "meta.csv") as f:
        meta = list(csv.DictReader(f))
    emb = np.load(mined / "emb.npy").astype(np.float32)
    split: dict[str, dict] = {s: dict(x=[], cls=[], fam=[], small=[]) for s in ("train", "test")}
    for r in meta:
        lab = labels.get(int(r["id"]), "U")
        if lab in ("R", "M", "S"):
            d = split[r["split"]]
            d["x"].append(emb[int(r["id"])])
            d["cls"].append(lab)
            d["fam"].append(family(r["video"]))
            d["small"].append(int(r["h"]) < 200)
    return {s: {k: np.array(v) for k, v in d.items()} for s, d in split.items()}


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    base = np.load(argv[1])
    x, cls, fam, src = base["X"].astype(np.float32), base["C"], base["G"], base["SRC"]  # file keys of the earlier rounds
    mined = load_mined(Path(argv[2]), Path(argv[3])) if len(argv) > 3 else None
    rep: dict = {"labeled_before": {c: int((cls == c).sum()) for c in "RMS"}}
    if mined is not None and len(mined["train"]["cls"]):
        tr = mined["train"]
        rep["hard_round"] = {c: int((tr["cls"] == c).sum()) for c in "RMS"}
        rep["before_lofo"] = report(lofo(x, cls, fam), cls, fam, src)  # old data only: the earlier heads' recipe
        x, cls, fam = np.concatenate([x, tr["x"]]), np.concatenate([cls, tr["cls"]]), np.concatenate([fam, tr["fam"]])
        src = np.concatenate([src, np.full(len(tr["cls"]), "hard")])
    p = lofo(x, cls, fam)
    rep["lofo"] = report(p, cls, fam, src)
    if mined is not None and "before_lofo" in rep:  # how the OLD recipe scores the new hard crops, family held out
        old_only = src != "hard"
        p_old = np.full(len(cls), np.nan)
        for g in sorted(set(fam) - {"old"}):
            test, train = fam == g, (fam != g) & old_only
            p_old[test] = prisk(fit(x[train], cls[train]), x[test])
        rep["lofo_old_recipe_on_all"] = report(p_old, cls, fam, src)
    thr = rep["lofo"]["at_0.85"]["thr"]
    heads = fit(x, cls)
    for name, text in ((FAST_FILE, _quantise(heads[0])), (MALE_FILE, _quantise(heads[1])), (THR_FILE, f"{thr:.4f}")):
        (out / name).write_text(text)
    if mined is not None and len(mined["test"]["cls"]):
        te = mined["test"]
        new = RiskHeads.load(out)
        rep["test_new"] = score_test(new.prisk(te["x"]), new.thr, te["cls"], te["small"])
        if len(argv) > 4:
            old = RiskHeads.load(argv[4])
            rep["test_old"] = score_test(old.prisk(te["x"]), old.thr, te["cls"], te["small"])
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print("HEADS_REPORT " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
