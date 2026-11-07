"""Run summary: timings / counters from the gate and the analyzer, written as one JSON line."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

FRAME_LISTS = ("failclosed_frames", "held_frames", "scene_frames", "muted_frames")  # kept as frame-id lists, not summarised
MAX_LISTED = 200


def summarize(gate_stats: dict, analyzer_stats: dict, n_frames: int, n_decisions: int, fps: float, extra: dict) -> dict[str, Any]:
    merged = {**gate_stats, **analyzer_stats}
    out: dict[str, Any] = {}
    for key, values in merged.items():
        if key in FRAME_LISTS:
            out[key] = list(values)[:MAX_LISTED]
        elif values and isinstance(values[0], (int, float)):
            a = np.asarray(values, np.float64)
            out[key] = dict(
                n=len(a),
                mean=round(float(a.mean()), 2),
                p50=round(float(np.percentile(a, 50)), 1),
                p95=round(float(np.percentile(a, 95)), 1),
                max=round(float(a.max()), 1),
            )
        else:
            out[key] = values
    seconds = n_frames / fps if fps else 0.0
    out["summary"] = dict(
        frames_out=n_frames,
        seconds=round(seconds, 1),
        decisions=n_decisions,
        decisions_per_s=round(n_decisions / max(1e-6, seconds), 2),
        failclosed_frames=len(merged.get("failclosed_frames", [])),
        held_frames=len(merged.get("held_frames", [])),
        scene_frames=len(merged.get("scene_frames", [])),
        muted_frames=len(merged.get("muted_frames", [])),
        analyzer_errors=len(merged.get("analyzer_errors", [])),
        **extra,
    )
    return out


def write_outputs(report: dict, stats_path: str, diag_path: str, decisions_log: list, draw_log: list | None) -> None:
    if stats_path:
        with open(stats_path, "w") as f:
            json.dump(report, f)
    if diag_path:
        with open(diag_path, "w") as f:
            json.dump(dict(dec=decisions_log, frm=draw_log or []), f)
    print("LIVE_RESULT " + json.dumps(report), flush=True)


def device_name() -> str:
    """GPU the models run on (``cpu`` when torch is not loaded in this process)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)
