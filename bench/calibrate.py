"""Fair thresholds for head recipes: one leave-one-game-family-out pass over ALL labeled game crops
(the earlier rounds + both mined rounds), threshold at 85% women recall over that whole population,
the same convention the current heads were calibrated with (compare_heads).

Why: bench.pose_eval sets the threshold on the mined crops alone. Those are the hardest crops by
construction, so 85% recall there forces a low threshold and the heads then cover far more junk in real
videos (run 39 disagreement sheets: blurred crowds, bikes, faces). Here every recipe is also compared at
equal modest-covered rate (the same false-cover budget as today's heads), which shows what it really buys.

usage: python -m bench.calibrate OUT_DIR LABELED.npz REFIND.npz
Writes OUT_DIR/<recipe>_cal/{fast_h6,male_h6,thr_h6}.txt and OUT_DIR/calibrate.json; prints "CAL <name> <json>".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from bench.pose_eval import fit, logits, prisk_from, score
from bench.train_heads import FAST_FILE, MALE_FILE, THR_FILE, _quantise, family


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    lab, rf = np.load(argv[1]), np.load(argv[2])
    keep = (lab["SRC"] != "aug") & (lab["C_new"] != "-")
    lx, lc, lf = lab["X"][keep].astype(np.float32), lab["C_new"][keep], lab["G"][keep]
    tag, split, cls = rf["tag"], rf["split"], rf["label"]
    mf = np.array([family(v) for v in rf["video"]])
    xh = rf["x_high"].astype(np.float32)
    tr, te = split == "train", split == "test"
    # one table: labeled rows then mined training rows (full-res)
    x = np.concatenate([lx, xh[tr]])
    fam = np.concatenate([lf, mf[tr]])
    c_all = np.concatenate([lc, cls[tr]])
    is_m = np.concatenate([np.zeros(len(lc), bool), np.ones(int(tr.sum()), bool)])
    m_tag = np.concatenate([np.full(len(lc), ""), tag[tr]])
    recipes = {  # which rows train: labeled always; mined rows by round; N as safe
        "ali": (~is_m | ((m_tag == "r30") & np.isin(c_all, ("R", "S", "M")))),
        "ali32": (~is_m | np.isin(c_all, ("R", "S", "M"))),
        "ali32n": np.ones(len(c_all), bool),
    }
    train_cls = np.where(c_all == "N", "S", c_all)
    scored = (fam != "old") & np.isin(c_all, ("R", "S", "M", "N"))
    rep: dict = {}
    base_modest = None
    for name, rows in recipes.items():
        p = np.full(len(c_all), np.nan)
        for g in sorted(set(fam[scored])):
            m = rows & (fam != g)
            h = fit(x[m], train_cls[m])
            t = (fam == g) & scored
            p[t] = prisk_from(logits(h, x[t]))
        s = scored
        r = {
            "lofo_all": score(p[s], c_all[s]),
            "lofo_mined": score(p[s & is_m], c_all[s & is_m]),
            "lofo_labeled": score(p[s & ~is_m], c_all[s & ~is_m]),
        }
        thr = r["lofo_all"]["thr"]
        if base_modest is None:
            base_modest = r["lofo_all"]["modest"]
        # equal false-cover budget: the threshold where modest covered = today's heads
        mod = np.sort(p[s & (c_all == "S")])
        thr_eq = float(mod[round((1 - base_modest) * (len(mod) - 1))])
        r["lofo_all_eq_modest"] = score(p[s], c_all[s], thr_eq)
        h = fit(x[rows], train_cls[rows])
        pt = prisk_from(logits(h, xh[te]))
        r["test"] = score(pt, cls[te], thr)
        r["test_eq_modest"] = score(pt, cls[te], thr_eq)
        for suffix, t in (("_cal", thr), ("_eq", thr_eq)):
            d = out / f"{name}{suffix}"
            d.mkdir(parents=True, exist_ok=True)
            for fname, text in ((FAST_FILE, _quantise(h[0])), (MALE_FILE, _quantise(h[1])), (THR_FILE, f"{t:.4f}")):
                (d / fname).write_text(text)
        rep[name] = r
        print("CAL", name, json.dumps(r), flush=True)
    (out / "calibrate.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
