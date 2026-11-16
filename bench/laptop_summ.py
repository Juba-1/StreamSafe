"""One table row per realtime run of the laptop-class GPU job (``bench/laptop_job.sh``).

usage: python -m bench.laptop_summ RESULT_DIR [NAME ...]        (reads NAME_st.json, NAME_diag.json, NAME_util.csv)

Columns: slate episodes and seconds (what the viewer sees), decisions per second and their time, the gap between
decisions in frames (9 is the slate limit), how old the garment shape on screen is (frames from the decision it
came from: the stutter a viewer sees), the share of garment draws that could not follow the motion, and what the
budget and the governor did.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

COLUMNS = [
    "run",
    "gpu",
    "util%",
    "dec/s",
    "slates",
    "slate_s",
    "held",
    "ms_p50",
    "ms_p95",
    "ms_max",
    "eye_p95",
    "garm_p95",
    "gap_p50",
    "gap_p95",
    "gap_max",
]
COLUMNS += ["age_mean", "age_p95", "nomotion%", "looks_max", "budget0", "budget_min", "lowered", "plain%", "put_off", "covered"]


def episodes(frames: list[int]) -> int:
    frames = sorted(frames)
    return sum(1 for i, v in enumerate(frames) if i == 0 or v - frames[i - 1] > 1)


def q(report: dict, key: str, what: str):
    value = report.get(key)
    return value.get(what) if isinstance(value, dict) else None


def row(folder: Path, name: str) -> dict:
    r = json.loads((folder / f"{name}_st.json").read_text())
    s = r["summary"]
    out: dict = dict(run=name, gpu=s.get("gpu"))
    util = folder / f"{name}_util.csv"
    if util.is_file():
        values = [float(line.split(",")[0]) for line in util.read_text().splitlines() if line.strip()[:1].isdigit()]
        out["util%"] = round(float(np.mean(values))) if values else None
    listed = r.get("failclosed_frames") or []
    out["dec/s"] = s.get("decisions_per_s")
    out["slates"] = s.get("slate_episodes", episodes(listed) if len(listed) == s.get("failclosed_frames") else f">={episodes(listed)}")
    out["slate_s"] = round(s.get("failclosed_frames", 0) / 30, 1)
    out["held"] = s.get("held_frames")
    out.update(ms_p50=q(r, "ms_total", "p50"), ms_p95=q(r, "ms_total", "p95"), ms_max=q(r, "ms_total", "max"))
    out.update(eye_p95=q(r, "ms_eye", "p95"), garm_p95=q(r, "ms_garment", "p95"), looks_max=q(r, "embedded", "max"))
    head = s.get("headroom") or {}
    out["budget0"] = head.get("budget_applied", "-")
    gov = (r.get("governor") or [{}])[0] if isinstance(r.get("governor"), list) else {}
    out["budget_min"] = gov.get("budget_lowest", "-")
    out["lowered"] = gov.get("budget_lowered", "-")
    out["plain%"] = round(100 * q(r, "plain_covers", "mean"), 1) if isinstance(r.get("plain_covers"), dict) else "-"
    out["put_off"] = q(r, "eye_put_off", "mean")
    out["covered"] = q(r, "risky", "mean")
    diag = folder / f"{name}_diag.json"
    if diag.is_file():
        d = json.loads(diag.read_text())
        decisions = sorted(row[0] for row in d.get("dec", []))
        gaps = np.diff(decisions) if len(decisions) > 1 else np.array([0])
        out.update(gap_p50=float(np.percentile(gaps, 50)), gap_p95=float(np.percentile(gaps, 95)), gap_max=int(gaps.max()))
        frm = np.asarray(d.get("frm") or np.zeros((0, 9)), np.float64)
        if len(frm):
            age = np.abs(frm[:, 0] - frm[:, 2])
            out.update(age_mean=round(float(age.mean()), 2), age_p95=float(np.percentile(age, 95)))
            out["nomotion%"] = round(100 * float((frm[:, 7] == 0).mean()), 1)
    return out


def main(argv: list[str]) -> int:
    folder = Path(argv[0])
    names = argv[1:] or sorted(p.name[: -len("_st.json")] for p in folder.glob("*_st.json"))
    rows = []
    for name in names:
        try:
            rows.append(row(folder, name))
        except (OSError, ValueError, KeyError) as exc:
            print(f"SUMM_SKIP {name}: {exc!r:.200}")
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in COLUMNS) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
