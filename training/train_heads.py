"""Train the gated risk heads (the recipe of runs 16-21) on every labeled crop, optionally plus a
newly labeled hard-mining round, and report how they generalise to games they never saw.

The recipe (gated heads, LOFO threshold) is described in training/common.py.

usage: python -m training.train_heads OUT_DIR LABELED_EMB.npz [MINED_DIR LABELS.csv [OLD_HEAD_DIR]]
LABELS.csv: id,label with label R (revealing woman), M (revealing man), S (safe), N (not a person), U (unsure);
only R/M/S are used. The mined "test" split (held-out videos) is never trained on; it is scored with the
new heads and, when OLD_HEAD_DIR is given, with the old ones.
Writes OUT_DIR/{fast_h6,male_h6,thr_h6}.txt (the engine's head files) and OUT_DIR/report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from streamsafe.models.heads import RiskHeads
from training.common import fit, load_mined, lofo, prisk, report, score_test, write_heads


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
    write_heads(out, fit(x, cls), thr)
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
