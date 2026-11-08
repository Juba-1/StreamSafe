"""One summary line per engine run, read from the LIVE_RESULT line on stdin.
usage: python -m streamsafe ... | python -m bench.summ NAME"""

import json
import sys


def stat(report: dict, key: str, q: str = "mean"):
    value = report.get(key)
    return value.get(q) if isinstance(value, dict) else None


def vram(report: dict):
    v = report.get("vram_gb")
    if isinstance(v, dict):
        return v.get("max")
    return v[0] if isinstance(v, list) and v else None


def line(name: str, r: dict) -> str:
    s = r["summary"]
    fields = [
        ("dps", s.get("decisions_per_s")),
        ("slates", s.get("failclosed_frames")),
        ("held", s.get("held_frames")),
        ("late%", round(100 * (stat(r, "late_frames") or 0), 2)),
        ("render95", stat(r, "render_ms", "p95")),
        ("analyze", stat(r, "ms_total")),
        ("analyze95", stat(r, "ms_total", "p95")),
        ("det", stat(r, "ms_detect")),
        ("eye", stat(r, "ms_eye")),
        ("pose", stat(r, "ms_pose")),
        ("garment", stat(r, "ms_garment")),
        ("people", stat(r, "people")),
        ("risky", stat(r, "risky")),
        ("vram", vram(r)),
        ("errors", s.get("analyzer_errors")),
        ("slate_at", (r.get("failclosed_frames") or [])[:12]),
    ]
    return " ".join(["RUN", name, str(s.get("gpu"))] + [f"{k} {v}" for k, v in fields])


if __name__ == "__main__":
    name = sys.argv[1]
    for raw in sys.stdin:
        if raw.startswith("LIVE_RESULT "):
            print(line(name, json.loads(raw[len("LIVE_RESULT ") :])), flush=True)
        elif "Error" in raw or "Traceback" in raw:
            print("ERR", name, raw.strip()[:300], flush=True)
