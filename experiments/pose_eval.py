"""Run 39: do Ali's review32 labels and posture features make the eye's heads better?

Head recipes (gated logistic heads, C=2 balanced) on the earlier labeled set + the mined rounds (training/refind.py):
    ali      earlier set + round-30 mined crops cut from the analysis frame (= today's heads)
    ali_hi   the same crops cut from the full-resolution frame (what the live engine sees)
    ali32    + round-38 mined crops (review32), full-res
    ali32n   + the not_person crops of both rounds as safe examples for the women head
Every recipe is scored on the same rows: the full-res mined training crops of both rounds, leave-one-game-family-out,
threshold = 85% women recall; then on the held-out test videos with its final heads.

Posture (streamsafe.analysis.posture) on top of the best eye recipe:
    stack    a small logistic model on [eye logits, posture features], fitted on out-of-fold eye logits
    rule_k   p x k when the pose is suggestive (lying / bent over / head below hips / legs wide apart / seen from behind)

usage: python -m experiments.pose_eval OUT_DIR LABELED.npz REFIND.npz
Writes OUT_DIR/<recipe>/{fast_h6,male_h6,thr_h6}.txt and OUT_DIR/report39.json; prints "P39 <name> <json>".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from streamsafe.analysis.posture import NAMES, features
from training.common import family, fit_strict, logistic, logits, prisk_from_logits, score, write_heads

INNER = 3  # inner folds for out-of-fold eye logits (stacking)


def posture_matrix(rf: np.lib.npyio.NpzFile) -> np.ndarray:
    box = rf["box"].astype(np.float32)
    aspect = (box[:, 2] - box[:, 0]) / np.maximum(1, box[:, 3] - box[:, 1])
    return np.stack([features(k, s, a) for k, s, a in zip(rf["kp"], rf["ks"], aspect, strict=True)])


def suggestive(f: np.ndarray) -> np.ndarray:
    col = {n: i for i, n in enumerate(NAMES)}
    return (
        (f[:, col["tilt"]] > 0.5)  # lying or bent over
        | (f[:, col["head_below_hips"]] > 0)
        | (f[:, col["aspect"]] > 1.1)  # wider than tall: lying / sprawled
        | (f[:, col["knee_spread"]] > 2.2)
        | (f[:, col["back_view"]] > 0)
    )


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    lab, rf = np.load(argv[1]), np.load(argv[2])
    keep = (lab["SRC"] != "aug") & (lab["C_new"] != "-")
    bx, bc = lab["X"][keep].astype(np.float32), lab["C_new"][keep]
    tag, split, cls = rf["tag"], rf["split"], rf["label"]
    fam = np.array([family(v) for v in rf["video"]])
    xh, xl = rf["x_high"].astype(np.float32), rf["x_low"].astype(np.float32)
    small = rf["h"] < 200
    post = posture_matrix(rf)
    tr, te = split == "train", split == "test"
    scored = tr & np.isin(cls, ("R", "S", "M"))
    rsm = np.isin(cls, ("R", "S", "M"))
    recipes = {  # name: mined rows used (mask), their embedding, N as safe
        "ali": (tr & rsm & (tag == "r30"), xl, False),
        "ali_hi": (tr & rsm & (tag == "r30"), xh, False),
        "ali32": (tr & rsm, xh, False),
        "ali32n": (tr, xh, True),
    }
    rep: dict = {
        "mined": {
            t: {s: {c: int(((tag == t) & (split == s) & (cls == c)).sum()) for c in "RSMN"} for s in ("train", "test")}
            for t in sorted(set(tag))
        },
        "posture_names": NAMES,
    }
    col = lambda use_n, m: np.where((cls == "N") & use_n, "S", cls)[m]  # noqa: E731
    best_lofo: dict[str, np.ndarray] = {}
    for name, (rows, xm, use_n) in recipes.items():
        p = np.full(len(cls), np.nan)
        lg_all = np.full((len(cls), 2), np.nan)
        for g in sorted(set(fam[scored])):
            m = rows & (fam != g)
            h = fit_strict(np.concatenate([bx, xm[m]]), np.concatenate([bc, col(use_n, m)]))
            t = (fam == g) & tr
            lg_all[t] = logits(h, xh[t])
            p[t] = prisk_from_logits(lg_all[t])
        best_lofo[name] = lg_all
        r = {"lofo": score(p[tr], cls[tr], small=small[tr])}
        h = fit_strict(np.concatenate([bx, xm[rows]]), np.concatenate([bc, col(use_n, rows)]))
        r["test"] = score(prisk_from_logits(logits(h, xh[te])), cls[te], r["lofo"]["thr"], small[te])
        write_heads(out / name, h, r["lofo"]["thr"])
        rep[name] = r
        print("P39", name, json.dumps(r), flush=True)

    # posture: what it says per class (training rows), and whether it adds to the eye
    sug = suggestive(post)
    rep["posture_means"] = {c: dict(zip(NAMES, np.round(post[tr & (cls == c)].mean(0), 3).tolist(), strict=True)) for c in "RSMN"}
    rep["suggestive_share"] = {c: round(float(sug[tr & (cls == c)].mean()), 4) for c in "RSMN"}
    base = "ali32n"
    rows, xm, use_n = recipes[base]
    lg = best_lofo[base]
    p_eye = prisk_from_logits(lg)
    for k in (1.5, 2.0, 3.0):
        pk = np.where(sug, np.minimum(1.0, p_eye * k), p_eye)
        r = {"lofo": score(pk[tr], cls[tr], small=small[tr])}
        h = fit_strict(np.concatenate([bx, xm[rows]]), np.concatenate([bc, col(use_n, rows)]))
        pt = prisk_from_logits(logits(h, xh[te]))
        r["test"] = score(np.where(sug[te], np.minimum(1.0, pt * k), pt), cls[te], r["lofo"]["thr"], small[te])
        rep[f"rule_{k}"] = r
        print("P39", f"rule_{k}", json.dumps(r), flush=True)
    # stacking, honestly nested: out-of-fold eye logits inside each outer fold
    zf = lambda lgx, f: np.concatenate([lgx, f], 1)  # noqa: E731
    p_st = np.full(len(cls), np.nan)
    fams = sorted(set(fam[scored]))
    for g in fams:
        m = rows & (fam != g)
        inner = [f for f in fams if f != g]
        oof = np.full((len(cls), 2), np.nan)
        for j in range(INNER):
            hold = np.isin(fam, inner[j::INNER])
            mi = m & ~hold
            hi = fit_strict(np.concatenate([bx, xm[mi]]), np.concatenate([bc, col(use_n, mi)]))
            sel = m & hold
            oof[sel] = logits(hi, xh[sel])
        st_rows = m & ~np.isnan(oof[:, 0])
        stk = logistic(1.0).fit_strict(zf(oof[st_rows], post[st_rows]), cls[st_rows] == "R")
        t = (fam == g) & tr
        p_st[t] = stk.predict_proba(zf(lg[t], post[t]))[:, 1]
    r = {"lofo": score(p_st[tr], cls[tr], small=small[tr])}
    stk = logistic(1.0).fit_strict(zf(lg[tr], post[tr]), cls[tr] == "R")  # final stacker on the LOFO logits of all training rows
    h = fit_strict(np.concatenate([bx, xm[rows]]), np.concatenate([bc, col(use_n, rows)]))
    r["test"] = score(stk.predict_proba(zf(logits(h, xh[te]), post[te]))[:, 1], cls[te], r["lofo"]["thr"], small[te])
    r["weights"] = dict(zip(["eye_women", "eye_male", *NAMES], np.round(stk.coef_[0], 3).tolist(), strict=True))
    rep["stack"] = r
    print("P39 stack", json.dumps(r), flush=True)
    (out / "report39.json").write_text(json.dumps(rep, indent=1))
    print("P39_REPORT " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
