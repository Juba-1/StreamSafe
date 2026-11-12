"""Which device does ONNX Runtime really reach in THIS process? ``<exe> gpu-probe`` prints one JSON object.

The app runs it once in NVIDIA mode after the NVIDIA libraries were installed (and before every start): a
session on a 111-byte model is opened the way the engine opens its models. ``device`` is where it really ran:
"cuda" means the NVIDIA runtime works; anything else and the app starts the engine on DirectML instead.
Listing a provider as "available" proves nothing: a missing DLL or an old driver only shows when a session opens.
"""

from __future__ import annotations

import json

# x[1,4] + (1,2,3,4) -> y, ONNX IR 8 / opset 13 (bytes written once with onnx.helper; no file is needed)
TINY_MODEL = bytes.fromhex(
    "0808120570726f62653a5e0a0e0a01780a01631201792203416464120570726f62652a1b08010804100122100000803f00000040"
    "00004040000080404201635a130a0178120e0a0c080112080a0208010a02080462130a0179120e0a0c080112080a0208010a0208"
    "0442040a00100d"
)


def probe() -> dict:
    out: dict = dict(onnxruntime="", file="", available=[], device="", error="", runtime="")
    try:
        import numpy as np
        import onnxruntime as ort

        from streamsafe.app import runtime_select
        from streamsafe.models.onnx_backend.session import device_of, providers

        out.update(onnxruntime=ort.__version__, file=str(ort.__file__), available=list(ort.get_available_providers()))
        out["runtime"] = runtime_select.NVIDIA if runtime_select.active() else runtime_select.DIRECTML
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False  # what DirectML needs; harmless elsewhere
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(TINY_MODEL, sess_options=opts, providers=providers())
        y = session.run(None, {"x": np.zeros((1, 4), np.float32)})[0]
        if y.tolist() != [[1.0, 2.0, 3.0, 4.0]]:
            raise RuntimeError(f"the test model answered {y.tolist()}")
        out["device"] = device_of(session)
    except Exception as exc:  # every failure is an answer here: "this runtime cannot be used"
        out["error"] = " ".join(f"{type(exc).__name__}: {exc}".split())[:400]
    return out


def main(argv: list[str] | None = None) -> int:
    result = probe()
    print("GPU_PROBE " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["device"] in ("cuda", "dml") else 1
