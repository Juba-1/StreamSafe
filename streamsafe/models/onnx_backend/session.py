"""One place that opens ONNX Runtime sessions, so every model runs on the same device.

Provider order: NVIDIA (CUDA), then any DirectX 12 GPU on Windows (DirectML: NVIDIA, AMD, Intel), then CPU.
``STREAMSAFE_EP`` forces one ("cuda", "dml", "cpu") for tests and the speed bench.

TensorRT (``TRT=1``, NVIDIA only) is never picked on its own: it goes in front of CUDA when the config asks for
it. It compiles each model into an engine for this GPU the first time (minutes) and loads it from ``trt_cache``
afterwards. What TensorRT cannot run stays on CUDA, so a model never leaves the GPU.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

LAST_DEVICE = ""  # device of the last session opened ("trt" / "cuda" / "dml" / "cpu"), for the run report
PROVIDERS = {
    "trt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
ORDER = ("cuda", "dml", "cpu")

# input name -> (smallest, usual, largest) shape, for models whose batch size varies
Shapes = dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]


@dataclass(frozen=True)
class Trt:
    """TensorRT settings for every session of one model folder."""

    cache: str  # folder the engines are kept in
    fp16: bool = True
    files: dict = field(default_factory=dict)  # manifest.json "files" (already verified): file name -> sha256 ...

    def folder(self, name: str) -> Path:
        """Engine folder of one model file. ONNX Runtime finds a cached engine by the model's name, not by its
        bytes, and says old engines must be deleted by hand when the model or the runtime changes. So the folder
        is named after both: the file's sha256 (with its weight file) and the ONNX Runtime version. An engine
        built from other weights is then simply not found, and a new one is built."""
        import onnxruntime as ort

        shas = "".join(meta["sha256"] for file, meta in sorted(self.files.items()) if file.startswith(name))
        tag = hashlib.sha256(shas.encode()).hexdigest()[:16]
        return Path(self.cache) / f"{Path(name).stem}-{tag}-ort{ort.__version__}{'-fp16' if self.fp16 else ''}"


def providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    forced = os.environ.get("STREAMSAFE_EP", "").strip().lower()
    if forced:
        if forced not in ORDER or PROVIDERS[forced] not in available:
            raise RuntimeError(f"STREAMSAFE_EP={forced!r} is not available here (have: {sorted(available)})")
        return [PROVIDERS[forced]]
    return [PROVIDERS[k] for k in ORDER if PROVIDERS[k] in available]


def trt_provider(trt: Trt, name: str, shapes: Shapes | None = None) -> tuple[str, dict]:
    """The TensorRT provider entry for one model file.

    ``shapes``: without them TensorRT builds the engine for the first batch size it meets and builds it AGAIN
    for every batch size outside what it has met (each time a pause of minutes: a slate). With them one engine
    covers the whole range from the start."""
    folder = trt.folder(name)
    folder.mkdir(parents=True, exist_ok=True)
    options: dict = {
        "trt_fp16_enable": trt.fp16,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(folder),
        "trt_timing_cache_enable": True,  # a rebuild (new driver, new ONNX Runtime) reuses the kernel timings
        "trt_timing_cache_path": str(trt.cache),
        "trt_layer_norm_fp32_fallback": True,  # a layer norm exported as Pow + Reduce ops overflows in fp16
    }
    for i, key in enumerate(("min", "opt", "max") if shapes else ()):
        options[f"trt_profile_{key}_shapes"] = ",".join(f"{inp}:{'x'.join(map(str, s[i]))}" for inp, s in shapes.items())
    return PROVIDERS["trt"], options


def open_session(path: str | Path, allow_cpu: bool = False, trt: Trt | None = None, shapes: Shapes | None = None):
    """An inference session on the best device. DirectML needs sequential execution and no memory pattern.
    No GPU and ``allow_cpu`` False -> RuntimeError: the engine never drops to the CPU silently (far too slow to
    protect a live stream; the system checker tells the streamer why).
    ``trt`` + ``shapes`` (the model's batch range): TensorRT in front of CUDA, see ``trt_provider``."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    chosen: list = providers()
    if chosen[0] == PROVIDERS["cpu"] and not allow_cpu:
        raise RuntimeError("no usable GPU for ONNX Runtime (CUDA / DirectML); set ALLOW_CPU=1 only for tests")
    if chosen[0] == PROVIDERS["dml"]:
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if trt is not None:
        if chosen[0] != PROVIDERS["cuda"] or PROVIDERS["trt"] not in ort.get_available_providers():
            raise RuntimeError("TRT=1 needs an NVIDIA GPU, onnxruntime-gpu and the TensorRT libraries (or set TRT=0)")
        chosen = [trt_provider(trt, Path(path).name, shapes), *chosen]
    session = ort.InferenceSession(str(path), sess_options=opts, providers=chosen)
    if device_of(session) == "cpu" and not allow_cpu:  # the GPU provider failed to initialise for this model
        raise RuntimeError(f"{Path(path).name} could not start on the GPU ({chosen[0]})")
    global LAST_DEVICE
    LAST_DEVICE = "onnx-" + device_of(session)  # "onnx-cuda" with TRT=1: TensorRT did not start, CUDA runs it
    return session


def device_of(session) -> str:
    """Which device a session really runs on ("trt", "cuda", "dml" or "cpu")."""
    first = session.get_providers()[0]
    return next(k for k, v in PROVIDERS.items() if v == first)
