"""One place that opens ONNX Runtime sessions, so every model runs on the same device.

Provider order: NVIDIA (CUDA), then any DirectX 12 GPU on Windows (DirectML: NVIDIA, AMD, Intel), then CPU.
``STREAMSAFE_EP`` forces one ("cuda", "dml", "cpu") for tests and the speed bench.
"""

from __future__ import annotations

import os
from pathlib import Path

LAST_DEVICE = ""  # device of the last session opened ("cuda" / "dml" / "cpu"), for the run report
PROVIDERS = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}
ORDER = ("cuda", "dml", "cpu")


def providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    forced = os.environ.get("STREAMSAFE_EP", "").strip().lower()
    if forced:
        if forced not in PROVIDERS or PROVIDERS[forced] not in available:
            raise RuntimeError(f"STREAMSAFE_EP={forced!r} is not available here (have: {sorted(available)})")
        return [PROVIDERS[forced]]
    return [PROVIDERS[k] for k in ORDER if PROVIDERS[k] in available]


def open_session(path: str | Path, allow_cpu: bool = False):
    """An inference session on the best device. DirectML needs sequential execution and no memory pattern.
    No GPU and ``allow_cpu`` False -> RuntimeError: the engine never drops to the CPU silently (far too slow to
    protect a live stream; the system checker tells the streamer why)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    chosen = providers()
    if chosen[0] == PROVIDERS["cpu"] and not allow_cpu:
        raise RuntimeError("no usable GPU for ONNX Runtime (CUDA / DirectML); set ALLOW_CPU=1 only for tests")
    if chosen[0] == PROVIDERS["dml"]:
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=opts, providers=chosen)
    if device_of(session) == "cpu" and not allow_cpu:  # the GPU provider failed to initialise for this model
        raise RuntimeError(f"{Path(path).name} could not start on the GPU ({chosen[0]})")
    global LAST_DEVICE
    LAST_DEVICE = "onnx-" + device_of(session)
    return session


def device_of(session) -> str:
    """Which device a session really runs on ("cuda", "dml" or "cpu")."""
    first = session.get_providers()[0]
    return next(k for k, v in PROVIDERS.items() if v == first)
