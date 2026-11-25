"""One place that opens ONNX Runtime sessions, so every model runs on the same device.

Provider order: NVIDIA (CUDA), then any DirectX 12 GPU on Windows (DirectML: NVIDIA, AMD, Intel), then CPU.
``STREAMSAFE_EP`` forces one ("cuda", "dml", "cpu") for tests and the speed bench.

TensorRT (``TRT=1``, NVIDIA only) is never picked on its own: it goes in front of CUDA when the config asks for
it. It compiles each model into an engine for this GPU (minutes) and loads it from the engine cache afterwards.
Which engine may be loaded is decided by ``engines.py``, never by ONNX Runtime. The chain for every model is:

    TensorRT  ->  CUDA  ->  refuse to start        (never the CPU)

A model that TensorRT cannot run (libraries missing, engine not prepared and building not allowed, a failed build,
an engine that answers with NaN) runs on CUDA, and the reason is kept (``OPENED``) for the start-up log and the run
report: a slower path is never taken silently. What TensorRT cannot run inside a model stays on CUDA too, so a
model never leaves the GPU.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from streamsafe.models.onnx_backend import engines

PROVIDERS = {
    "trt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
ORDER = ("cuda", "dml", "cpu")

# input name -> (smallest, usual, largest) shape, for models whose batch size varies
Shapes = dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]


class EngineError(RuntimeError):
    """A TensorRT engine could not be made ready where it had to be (``Trt.strict``: the prepare step)."""


@dataclass(frozen=True)
class Trt:
    """TensorRT settings for every session of one model folder."""

    cache: str  # folder the engines are kept in
    fp16: bool = True
    files: dict = field(default_factory=dict)  # manifest.json "files" (already verified): file name -> sha256 ...
    # may an engine that is not ready be built now? That takes minutes. False: CUDA runs the model and the report
    # says that the engines must be prepared (``python -m streamsafe.prepare``)
    build: bool = True
    strict: bool = False  # the prepare step: a model that does not end up on TensorRT is an error, not a CUDA session
    libs: str = ""  # folder with the TensorRT libraries; empty = the pip package or the system search path


@dataclass
class Opened:
    file: str
    device: str  # "trt" / "cuda" / "dml" / "cpu": where the model really runs
    note: str = ""  # why that is not the device asked for, or what happened to its engine at this start


OPENED: list[Opened] = []  # every session of this process, for the start-up log and the run report


def reset() -> None:
    OPENED.clear()


def device_summary() -> str:
    """ "onnx-trt", "onnx-cuda" ...; when the models are not all on one device every device is named
    ("onnx-trt+cuda"), so one model on a slower path shows in the run report."""
    devices = [d for d in PROVIDERS if any(o.device == d for o in OPENED)]
    return "onnx-" + "+".join(devices) if devices else ""


def notes() -> list[str]:
    return [f"{o.file}: {o.note}" for o in OPENED if o.note]


def providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    forced = os.environ.get("STREAMSAFE_EP", "").strip().lower()
    if forced:
        if forced not in ORDER or PROVIDERS[forced] not in available:
            raise RuntimeError(f"STREAMSAFE_EP={forced!r} is not available here (have: {sorted(available)})")
        return [PROVIDERS[forced]]
    return [PROVIDERS[k] for k in ORDER if PROVIDERS[k] in available]


def with_options(chosen: list[str]) -> list:
    """Provider list with per-provider options. CUDA: how cuDNN picks its convolution kernels.
    ``CUDNN_SEARCH`` = EXHAUSTIVE (ONNX Runtime's default: every kernel is timed at each new input shape),
    HEURISTIC or DEFAULT (no timing). Set for the load benchmark (tools/load_bench.py)."""
    search = os.environ.get("CUDNN_SEARCH", "").strip().upper()
    if search not in ("EXHAUSTIVE", "HEURISTIC", "DEFAULT"):
        return chosen
    return [(p, {"cudnn_conv_algo_search": search}) if p == PROVIDERS["cuda"] else p for p in chosen]


def trt_options(trt: Trt, shapes: Shapes | None = None) -> dict:
    """The TensorRT build options of one model: everything here changes the engine, so all of it is part of the
    engine key (``engines.engine_key``). The cache folder is added per session (``_trt_session``).

    ``shapes``: without them TensorRT builds the engine for the first batch size it meets and builds it AGAIN
    for every batch size outside what it has met (each time a pause of minutes: a slate). With them one engine
    covers the whole range from the start."""
    options: dict = {
        "trt_fp16_enable": trt.fp16,
        "trt_layer_norm_fp32_fallback": True,  # a layer norm exported as Pow + Reduce ops overflows in fp16
    }
    for i, key in enumerate(("min", "opt", "max") if shapes else ()):
        options[f"trt_profile_{key}_shapes"] = ",".join(f"{inp}:{'x'.join(map(str, s[i]))}" for inp, s in shapes.items())
    return options


def open_session(path: str | Path, allow_cpu: bool = False, trt: Trt | None = None, shapes: Shapes | None = None):
    """An inference session on the best device. DirectML needs sequential execution and no memory pattern.
    No GPU and ``allow_cpu`` False -> RuntimeError: the engine never drops to the CPU silently (far too slow to
    protect a live stream; the system checker tells the streamer why).
    ``trt`` + ``shapes`` (the model's batch range): TensorRT first, CUDA when TensorRT cannot run the model."""
    import onnxruntime as ort

    path = Path(path)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    chosen: list = providers()
    if chosen[0] == PROVIDERS["cpu"] and not allow_cpu:
        raise RuntimeError("no usable GPU for ONNX Runtime (CUDA / DirectML); set ALLOW_CPU=1 only for tests")
    if chosen[0] == PROVIDERS["dml"]:
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session, note = None, ""
    if trt is not None:
        if chosen[0] != PROVIDERS["cuda"]:
            raise RuntimeError("TRT=1 needs an NVIDIA GPU and onnxruntime-gpu (CUDA is the path behind TensorRT); or set TRT=0")
        session, note = _open_trt(ort, path, opts, trt, shapes)
        if session is None:
            if trt.strict:
                raise EngineError(f"{path.name}: {note}")
            note = f"TensorRT not used ({note}); running on CUDA"
    if session is None:
        session = ort.InferenceSession(str(path), sess_options=opts, providers=with_options(chosen))
    device = device_of(session)
    if device == "cpu" and not allow_cpu:  # the GPU provider failed to initialise for this model
        raise RuntimeError(f"{path.name} could not start on the GPU ({chosen[0]})")
    OPENED.append(Opened(path.name, device, note))
    return session


def _open_trt(ort, path: Path, opts, trt: Trt, shapes: Shapes | None):
    """(session on TensorRT, note about its engine) or (None, why TensorRT cannot run this model now).

    A sealed engine that fits this PC is loaded. One that does not load after all is deleted and, like a missing
    or stale one, built again if ``trt.build`` allows: into a private folder, tested, then sealed and moved into
    place (``engines``). At most one load and one build per call."""
    if PROVIDERS["trt"] not in ort.get_available_providers():
        return None, "this ONNX Runtime build has no TensorRT provider"
    engines.expose_libraries(trt.libs)
    try:
        ident = engines.identity(trt.libs)
    except engines.NotUsableError as exc:
        return None, str(exc)
    options = trt_options(trt, shapes)
    key = engines.engine_key(path.name, engines.model_shas(path, trt.files), options, ident)
    final = engines.folder(trt.cache, path.name, key)
    why = engines.why_not_ready(trt.cache, path.name, key)
    if not why:
        before = engines.listing(final)
        session, err = _trt_session(ort, path, opts, options, final, shapes)
        if session is not None:
            if engines.listing(final) == before:
                return session, ""
            # ONNX Runtime did not take the sealed engine and wrote its own: fine for this run (built from the
            # verified model just now, and tested), but the folder no longer matches its seal
            try:
                engines.seal(final, key)
            except OSError as exc:
                engines.remove(final)
                return session, f"the stored engine was not used and the new one could not be sealed ({exc})"
            return session, "the stored engine was not used: built again at this start"
        why = f"the stored engine did not load ({err})"
        engines.remove(final)
    if not trt.build:
        return None, f"engine not ready: {why}; prepare it with python -m streamsafe.prepare"
    try:
        work = engines.begin(final)
    except OSError as exc:
        return None, f"the engine folder {trt.cache} cannot be written ({exc})"
    start = time.perf_counter()
    session, err = _trt_session(ort, path, opts, options, work, shapes)
    if session is not None and not any(name.endswith(".engine") for name in engines.listing(work)):
        # the provider started but compiled nothing: the model would run on CUDA under TensorRT's name
        session, err = None, "TensorRT wrote no engine file: it took no part of the model, or the disk is full"
    if session is None:
        engines.remove(work)
        return None, f"the engine could not be built ({err})"
    stored = engines.commit(work, final, key)
    return session, f"engine built in {time.perf_counter() - start:.0f} s ({why})" + (f"; {stored}" if stored else "")


def _trt_session(ort, path: Path, opts, options: dict, folder: Path, shapes: Shapes | None):
    """(session, "") when the model really runs on TensorRT and answers a test input with finite numbers,
    otherwise (None, why).

    ONNX Runtime's own reaction to a TensorRT error is to retry on other providers and only print the error
    (1.22.0 retries on the CPU alone). That session is never used: the caller opens CUDA itself."""
    entry = (
        PROVIDERS["trt"],
        {
            **options,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(folder),
            # kernel timings of a build are kept with its engine: a shared file could be written by two builds at once
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(folder),
        },
    )

    class Recording(ort.InferenceSession):
        """Keeps the error of the first attempt, which ONNX Runtime prints and then drops when it retries."""

        ep_error = ""

        def _create_inference_session(self, *args, **kwargs):
            try:
                super()._create_inference_session(*args, **kwargs)
            except Exception as exc:
                self.ep_error = self.ep_error or str(exc)
                raise

    session, err = None, ""
    try:
        # enable_fallback=0: newer ONNX Runtime then raises instead of retrying (older ones ignore the argument)
        session = Recording(str(path), sess_options=opts, providers=[entry, PROVIDERS["cuda"]], enable_fallback=0)
        if device_of(session) != "trt":
            err = session.ep_error or "the TensorRT provider did not start (libraries missing or not the version this ONNX Runtime needs)"
        else:
            err = _selftest(session, shapes)
    except Exception as exc:
        err = str(exc) or type(exc).__name__
    if err:
        del session  # frees its GPU memory before CUDA (or a rebuild) loads the model again
        return None, " ".join(err.split())[:300]
    return session, ""


def _selftest(session, shapes: Shapes | None) -> str:
    """Run one fixed input through a fresh TensorRT session; "" when every output is finite.

    An fp16 engine that overflows answers with inf / NaN. Every comparison with NaN is false, so such a detector
    would find nobody and such an eye would call everybody safe: this engine must not be used at all."""
    feeds = {}
    for inp in session.get_inputs():
        shape = shapes[inp.name][0] if shapes and inp.name in shapes else tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
        x = np.random.default_rng(0).random(shape, np.float32)
        feeds[inp.name] = x.astype(np.float16) if "float16" in inp.type else x
    bad = [out.name for out, value in zip(session.get_outputs(), session.run(None, feeds), strict=True) if not np.isfinite(value).all()]
    return f"the engine's test run gave non-finite numbers in {bad}" if bad else ""


def device_of(session) -> str:
    """Which device a session really runs on ("trt", "cuda", "dml" or "cpu")."""
    first = session.get_providers()[0]
    return next(k for k, v in PROVIDERS.items() if v == first)
