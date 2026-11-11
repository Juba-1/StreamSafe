"""Run summary: timings / counters from the gate and the analyzer, written as one JSON line."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

# kept as frame-id lists, not summarised
FRAME_LISTS = ("failclosed_frames", "held_frames", "scene_frames", "muted_frames", "dropped_frames")
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
    """What the models run on: the CUDA device name (PyTorch), or the ONNX Runtime device(s): "onnx-trt",
    "onnx-cuda", "onnx-dml"; "onnx-trt+cuda" when some model is not on TensorRT. Otherwise "cpu"."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    session = sys.modules.get("streamsafe.models.onnx_backend.session")
    if session is not None and session.device_summary():
        return session.device_summary()
    return "cpu"


def device_notes() -> list[str]:
    """Per model: why it is not on the device that was asked for (TensorRT fell back to CUDA), or what happened
    to its engine at this start (built, rebuilt). Empty when every model started on its fast path from the cache."""
    session = sys.modules.get("streamsafe.models.onnx_backend.session")
    return session.notes() if session is not None else []


def device_stats(stats: dict) -> None:
    """Add the device facts of this process to an analyzer's stats (the analyzer may live in a worker process)."""
    stats["device"] = [device_name()]
    stats["device_notes"] = device_notes()


def device_summary(stats: dict) -> dict:
    """Take the device facts out of an analyzer's stats, as fields of the run summary:
    ``gpu``, ``device_notes`` and ``headroom`` (the model times measured at start, ``analysis/headroom.py``)."""
    return dict(
        gpu=stats.pop("device", ["?"])[0],
        device_notes=list(stats.pop("device_notes", [])),
        headroom=next(iter(stats.pop("headroom", [])), None),
    )
