"""What every head-training script shares: game families, the gated head recipe, scoring, the engine's head files.

The eye's risk score for one person crop x (a C-RADIOv4-H embedding) comes from two logistic heads:

    p(cover) = women(x) * (1 - male(x))

    women   revealing woman (R) vs everyone who is not a man (S, and N when present)
    male    man (M) vs the rest: ``fit`` uses every non-S row (the production recipe, run 37 "ali");
            ``fit_strict`` uses R only (the recipe of runs 39-40, kept so their numbers stay reproducible)

Both are logistic regression with balanced classes. The operating threshold is the one that keeps 85% of
revealing women covered when every game family is held out in turn (leave-one-family-out, "LOFO"), so the
threshold is set on games the heads did not see.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from streamsafe.models.heads import FAST_FILE, MALE_FILE, THR_FILE

RECALLS = (0.85, 0.90)  # women recall levels reported; the first one sets the threshold
FAMILIES = {  # clip-name prefix -> game family; anything else is GTA
    "cyber": "cyber", "yakuza": "yakuza", "sr22": "saintsrow", "sr3": "saintsrow", "sims4": "sims4", "tlou2": "tlou2", "wd2": "watchdogs",
    "wdl": "watchdogs", "sdogs": "sleepingdogs", "di2": "deadisland", "fc": "farcry", "hitman": "hitman", "rdr2": "rdr2",
    "lizzie": "lizzie",
}  # fmt: skip

Heads = tuple[LogisticRegression, LogisticRegression]  # (women, male)


def family(video: str) -> str:
    return next((name for prefix, name in FAMILIES.items() if video.startswith(prefix)), "gta")


def logistic(c: float = 2.0) -> LogisticRegression:
    return LogisticRegression(max_iter=5000, C=c, class_weight="balanced")


def fit(x: np.ndarray, cls: np.ndarray, w: np.ndarray | None = None, c: float = 2.0) -> Heads:
    """The production recipe. ``w``: optional per-row sample weight (photos, added examples)."""
    w = np.ones(len(cls)) if w is None else w
    a, b = cls != "M", cls != "S"
    return (
        logistic(c).fit(x[a], cls[a] == "R", sample_weight=w[a]),
        logistic(c).fit(x[b], cls[b] == "M", sample_weight=w[b]),
    )


def fit_strict(x: np.ndarray, cls: np.ndarray) -> Heads:
    """Runs 39-40: the male head sees M vs R only (S and N never enter it)."""
    a, b = cls != "M", np.isin(cls, ("M", "R"))
    return logistic().fit(x[a], cls[a] == "R"), logistic().fit(x[b], cls[b] == "M")


def prisk(heads: Heads, x: np.ndarray) -> np.ndarray:
    return heads[0].predict_proba(x)[:, 1] * (1 - heads[1].predict_proba(x)[:, 1])


def logits(heads: Heads, x: np.ndarray) -> np.ndarray:
    return np.stack([heads[0].decision_function(x), heads[1].decision_function(x)], 1)


def prisk_from_logits(lg: np.ndarray) -> np.ndarray:
    sig = 1 / (1 + np.exp(-lg))
    return sig[:, 0] * (1 - sig[:, 1])


def lofo(x: np.ndarray, cls: np.ndarray, fam: np.ndarray) -> np.ndarray:
    """Out-of-family p for every crop of a known family ('old' crops are training-only)."""
    p = np.full(len(cls), np.nan)
    for g in sorted(set(fam) - {"old"}):
        test = fam == g
        p[test] = prisk(fit(x[~test], cls[~test]), x[test])
    return p


def share(mask: np.ndarray, hit: np.ndarray) -> float | None:
    return round(float(hit[mask].mean()), 4) if mask.any() else None


def report(p: np.ndarray, cls: np.ndarray, fam: np.ndarray, src: np.ndarray) -> dict:
    """LOFO scores at each recall level: threshold, and what it covers per class / per data source."""
    known = fam != "old"
    women = known & (cls == "R")
    out = {}
    for rec in RECALLS:
        thr = float(np.quantile(p[women], 1 - rec))
        hit = p >= thr
        out[f"at_{rec}"] = dict(
            thr=round(thr, 4),
            modest_covered=share(known & (cls == "S"), hit),
            men_covered=share(known & (cls == "M"), hit),
            video_women_caught=share(women & np.isin(src, ["video", "video2"]), hit),
            hard_women_caught=share(women & (src == "hard"), hit),
            hard_modest_covered=share(known & (cls == "S") & (src == "hard"), hit),
        )
    return out


def score_test(p: np.ndarray, thr: float, cls: np.ndarray, small: np.ndarray) -> dict:
    """Held-out test videos at a fixed threshold."""
    hit = p >= thr
    return dict(
        thr=round(thr, 4),
        women_caught=share(cls == "R", hit),
        small_mid_women_caught=share((cls == "R") & small, hit),
        modest_covered=share(cls == "S", hit),
        men_covered=share(cls == "M", hit),
        n=dict(R=int((cls == "R").sum()), S=int((cls == "S").sum()), M=int((cls == "M").sum())),
    )


def score(p: np.ndarray, cls: np.ndarray, thr: float | None = None, small: np.ndarray | None = None, recall: float = 0.85) -> dict:
    """Threshold at ``recall`` women recall (unless given) and what it covers per class (runs 39-40 format)."""
    r = cls == "R"
    if thr is None:
        thr = float(np.quantile(p[r], 1 - recall)) if r.any() else 0.5
    hit = p >= thr
    out: dict = {"thr": round(thr, 4), "n": {c: int((cls == c).sum()) for c in "RSMN"}}
    for name, c in (("women", "R"), ("modest", "S"), ("men", "M"), ("not_person", "N")):
        if (cls == c).any():
            out[name] = round(float(hit[cls == c].mean()), 4)
    known = np.isin(cls, ("R", "S", "M"))
    if len(set(r[known])) == 2:
        out["auc"] = round(float(roc_auc_score(r[known], p[known])), 4)
    if small is not None and (r & small).any():
        out["women_small"] = round(float(hit[r & small].mean()), 4)
    return out


def quantise(head: LogisticRegression) -> str:
    """One head as the engine reads it (streamsafe/models/heads.py): scale;intercept;int16 weights."""
    w = head.coef_.ravel()
    scale = 32000 / np.abs(w).max()
    return f"{scale};{head.intercept_[0]:.5f};" + ",".join(str(round(v)) for v in w * scale)


def write_heads(out: Path, heads: Heads, thr: float) -> None:
    """The three files the engine loads from HEAD_DIR."""
    out.mkdir(parents=True, exist_ok=True)
    for name, text in ((FAST_FILE, quantise(heads[0])), (MALE_FILE, quantise(heads[1])), (THR_FILE, f"{thr:.4f}")):
        (out / name).write_text(text)


def load_mined(mined: Path, labels_csv: Path) -> dict[str, dict]:
    """A labeled hard-mining round (mine_hard.py output + Ali's labels): R/M/S rows per split (train / test)."""
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
