"""A stand-in for ONNX Runtime's session class on an NVIDIA PC, for the TensorRT tests (no GPU, no TensorRT).

It behaves like the real one where the engine cache depends on it: asked for TensorRT, it writes an engine file
into the cache folder it was given when there is none (a build) and reads it when there is one (a load). ``trt``
picks what TensorRT does:

  "ok"        builds / loads
  "fail"      raises (a failed build; newer ONNX Runtime with enable_fallback=0)
  "fail_cpu"  fails and comes back on the CPU alone, as ONNX Runtime 1.22.0 does by itself
  "skipped"   the provider does not start (libraries missing): ONNX Runtime goes on with CUDA without an error
  "nan"       builds an engine that answers with NaN
  "noengine"  starts but writes no engine file
  "ignore"    does not take the stored engine and writes another one
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from streamsafe.models.onnx_backend import engines, session

TRT, CUDA, DML, CPU = (session.PROVIDERS[k] for k in ("trt", "cuda", "dml", "cpu"))
ENGINE = "TensorrtExecutionProvider_TRTKernel_graph_fp16.engine"
GPU = (("NVIDIA GeForce RTX 3090", "561.09"),)


class FakeRuntime:
    def __init__(self) -> None:
        self.trt = "ok"
        self.cuda = True  # False: the CUDA provider fails to initialise, ONNX Runtime ends on the CPU
        self.asked: list[list] = []  # the provider list of every session created
        self.built = self.loaded = 0
        self.identity = engines.Identity(GPU, "1.22.0", "tensorrt-cu12-libs 10.9.0.34")
        runtime = self

        class Session:
            def __init__(self, path, sess_options=None, providers=None, **kwargs):
                try:
                    self._create_inference_session(providers, None)
                except RuntimeError:
                    if runtime.trt != "fail_cpu":
                        raise
                    self._create_inference_session([CPU], None)  # ONNX Runtime 1.22.0: retry on the CPU, error only printed

            def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
                runtime.asked.append(list(providers))
                names = [p if isinstance(p, str) else p[0] for p in providers]
                self.nan = False
                if names[0] == TRT:
                    if runtime.trt in ("fail", "fail_cpu"):
                        raise RuntimeError("TensorRT EP could not build engine for fused node: out of memory")
                    if runtime.trt == "skipped":
                        names = names[1:]
                    else:
                        self._engine(Path(providers[0][1]["trt_engine_cache_path"]))
                if names[0] == CUDA and not runtime.cuda:
                    names = [CPU]
                self.names = names

            def _engine(self, folder: Path) -> None:
                self.nan = runtime.trt == "nan"
                if runtime.trt == "noengine":
                    return
                if (folder / ENGINE).exists() and runtime.trt != "ignore":
                    runtime.loaded += 1
                    return
                runtime.built += 1
                name = ENGINE if runtime.trt != "ignore" else f"other{runtime.built}.engine"
                (folder / name).write_bytes(b"engine %d" % runtime.built)
                (folder / "TensorrtExecutionProvider_cache.timing").write_bytes(b"timings")

            def get_providers(self):
                return self.names

            def get_inputs(self):
                return [SimpleNamespace(name="pixels", shape=["n", 3, 8, 8], type="tensor(float)")]

            def get_outputs(self):
                return [SimpleNamespace(name="summary")]

            def run(self, _names, feeds):
                assert feeds["pixels"].dtype == np.float32 and feeds["pixels"].ndim == 4
                return [np.full((len(feeds["pixels"]), 3), np.nan if self.nan else 0.5, np.float32)]

        self.Session = Session

    def trt_asks(self) -> int:
        return sum(1 for p in self.asked if not isinstance(p[0], str))


def install(monkeypatch, available=(TRT, CUDA, CPU)) -> FakeRuntime:
    """Replace ONNX Runtime's session class and this PC's identity; returns the handle that steers them."""
    import onnxruntime as ort

    runtime = FakeRuntime()
    monkeypatch.setattr(ort, "get_available_providers", lambda: list(available))
    monkeypatch.setattr(ort, "InferenceSession", runtime.Session)
    monkeypatch.setattr(engines, "identity", lambda libs="": runtime.identity)
    monkeypatch.setattr(engines, "expose_libraries", lambda libs="": None)
    monkeypatch.delenv("STREAMSAFE_EP", raising=False)
    session.reset()
    return runtime
