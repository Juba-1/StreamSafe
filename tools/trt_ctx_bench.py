"""Experiment: can TensorRT sessions open faster? (GPU pod, after the prepare step)
  1. a normal start records every TensorRT session's model, options and engine folder
  2. for each model an "EP context" model is written (ONNX Runtime wraps the built engine: no TensorRT builder
     at load), then each way of opening is timed in a fresh process: normal, EP context, and all four at once.
Usage: python tools/trt_ctx_bench.py record OUT_DIR ; python tools/trt_ctx_bench.py time OUT_DIR MODE"""

import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TRT = "TensorrtExecutionProvider"


def record(out: Path) -> None:
    from streamsafe.config import Config
    from streamsafe.models import build_models
    from streamsafe.models.onnx_backend import session

    seen = []
    real = session._trt_session

    def spy(ort, path, opts, options, folder, shapes):
        seen.append({"model": str(path), "options": options, "folder": str(folder)})
        return real(ort, path, opts, options, folder, shapes)

    session._trt_session = spy
    build_models(Config.from_env(mode="realtime"))
    import onnxruntime as ort

    for item in seen:
        name = Path(item["model"]).stem
        ctx_dir = out / name
        shutil.rmtree(ctx_dir, ignore_errors=True)
        shutil.copytree(item["folder"], ctx_dir)  # the built engine, as the cache of the context model
        opts = {
            **item["options"],
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(ctx_dir),
            "trt_dump_ep_context_model": True,
            "trt_ep_context_file_path": str(ctx_dir),
            "trt_ep_context_embed_mode": 0,
        }
        before = set(ctx_dir.rglob("*"))
        s = ort.InferenceSession(item["model"], providers=[(TRT, opts), "CUDAExecutionProvider"])
        made = sorted(str(p) for p in set(ctx_dir.rglob("*")) - before)
        item["made"] = made
        item["ctx"] = [m for m in made if m.endswith(".onnx")]
        item["dump_provider"] = s.get_providers()[0]
    (out / "seen.json").write_text(json.dumps(seen, indent=1))
    print("RECORDED", json.dumps([(Path(s["model"]).name, s["dump_provider"], s["made"]) for s in seen]), flush=True)


def open_one(item: dict, mode: str) -> float:
    import onnxruntime as ort

    t = time.perf_counter()
    if mode == "ctx" and item.get("ctx"):
        folder = str(Path(item["ctx"][0]).parent)
        opts = {**item["options"], "trt_engine_cache_enable": True, "trt_engine_cache_path": folder,
                "trt_ep_context_file_path": folder}  # fmt: skip
        s = ort.InferenceSession(item["ctx"][0], providers=[(TRT, opts), "CUDAExecutionProvider"])
    else:
        opts = {**item["options"], "trt_engine_cache_enable": True, "trt_engine_cache_path": item["folder"],
                "trt_timing_cache_enable": True, "trt_timing_cache_path": item["folder"]}  # fmt: skip
        s = ort.InferenceSession(item["model"], providers=[(TRT, opts), "CUDAExecutionProvider"])
    took = time.perf_counter() - t
    assert s.get_providers()[0] == TRT, s.get_providers()
    return round(took, 2)


def timed(out: Path, mode: str) -> None:
    seen = json.loads((out / "seen.json").read_text())
    import onnxruntime  # noqa: F401  (import time is not what is measured)

    from streamsafe.config import Config
    from streamsafe.models.onnx_backend import checked_files, engines, trt_settings

    cfg = Config.from_env(mode="realtime")
    engines.expose_libraries(trt_settings(cfg, checked_files(cfg)).libs)  # as the engine does before TensorRT

    t = time.perf_counter()
    if mode.endswith("_parallel"):
        with ThreadPoolExecutor(len(seen)) as pool:
            each = list(pool.map(lambda it: open_one(it, mode.split("_")[0]), seen))
    else:
        each = [open_one(it, mode) for it in seen]
    print("CTXBENCH", json.dumps({"mode": mode, "total_s": round(time.perf_counter() - t, 2),
          "each": dict(zip([Path(s["model"]).name for s in seen], each, strict=True))}), flush=True)  # fmt: skip


if __name__ == "__main__":
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    record(out) if sys.argv[1] == "record" else timed(out, sys.argv[3])
