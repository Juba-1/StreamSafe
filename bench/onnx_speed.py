"""How fast does one exported model run on this GPU, per batch size and per way of running it? (GPU box)

    python -m bench.onnx_speed MODEL.onnx [BATCHES] [RUNS]        e.g.  ... models/eye.onnx 1,2,4,8,16 30

The session is opened exactly as the engine opens it, so the engine's settings pick what is measured:
``STREAMSAFE_EP`` (cuda / dml / cpu), ``TRT=1`` (+ ``TRT_FP16``, ``TRT_CACHE``), and the file itself
(eye.onnx vs eye16.onnx). Prints one line ``ONNX_SPEED <json>``:

  load_s      opening the session (with TRT=1: building the engine the first time, loading it afterwards)
  note        with TRT=1: what happened to the engine ("engine built in ...", "" = loaded from the cache), or why
              TensorRT was not used and CUDA runs the model (then ``device`` is "cuda")
  per batch size n:
    first_ms    the first run at this size: what a live stream pays if the size was not warmed up at start
    p50 / p95   steady runs, numpy in and out (how the engine calls it)
    per_item    p50 / n
    bound_p50   the same with input and output kept on the GPU (I/O binding; CUDA / TensorRT only): the gap to
                p50 is everything that keeping crops on the GPU could ever save
  switch_extra_ms  once all sizes were seen: how much longer than its p50 a run takes right after a run of another
              size (median). Clearly above 0 = the runtime prepares again on every size change, and padding
              batches to a few fixed sizes would pay off
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from streamsafe.config import Config
from streamsafe.models.onnx_backend.session import OPENED, Trt, device_of, open_session


def _ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return 1000 * (time.perf_counter() - start)


def _bound_runner(session, name: str, x: np.ndarray):
    """Run with input and output staying on the GPU, or None where that cannot be set up."""
    import onnxruntime as ort

    try:
        binding = session.io_binding()
        binding.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0))
        binding.bind_output(session.get_outputs()[0].name, "cuda", 0)
        session.run_with_iobinding(binding)
        return lambda: session.run_with_iobinding(binding)
    except Exception:
        return None


def measure(path: str | Path, batches: list[int], runs: int = 30, cfg: Config | None = None) -> dict:
    cfg = cfg or Config.from_env()
    path = Path(path)
    manifest = path.parent / "manifest.json"
    files = json.loads(manifest.read_text())["files"] if manifest.exists() else {}
    b = cfg.backend
    cache = b.trt_cache or str(path.parent / "trt_cache")
    trt = Trt(cache, b.trt_fp16, files, b.trt_build, libs=b.trt_libs) if b.trt else None

    shapes = None
    if trt is not None:  # the engine gives TensorRT its batch range too (models/onnx_backend/eye.py, pose.py)
        name, dims = _first_input(path)
        if not isinstance(dims[0], int):
            usual = sorted(batches)[len(batches) // 2]
            shapes = {name: tuple((n, *dims[1:]) for n in (min(batches), usual, max(batches)))}
    start = time.perf_counter()
    session = open_session(path, cfg.backend.allow_cpu, trt, shapes)
    load_s = time.perf_counter() - start
    inp = session.get_inputs()[0]
    fixed = isinstance(inp.shape[0], int)
    sizes = [int(inp.shape[0])] if fixed else batches
    rng = np.random.default_rng(0)
    feeds = {n: rng.random((n, *inp.shape[1:]), np.float32) for n in sizes}
    device = device_of(session)

    out: dict = dict(model=path.name, device=device, providers=session.get_providers(), load_s=round(load_s, 2), batch={})
    out["note"] = OPENED[-1].note  # TRT=1: "engine built in ..." the first time, "" when loaded from the cache, or why CUDA runs it
    for n, x in feeds.items():
        first = _ms(lambda x=x: session.run(None, {inp.name: x}))
        steady = sorted(_ms(lambda x=x: session.run(None, {inp.name: x})) for _ in range(runs))
        row = dict(first_ms=round(first, 1), p50=round(steady[len(steady) // 2], 2), p95=round(steady[int(0.95 * (len(steady) - 1))], 2))
        row["per_item"] = round(row["p50"] / n, 2)
        bound = _bound_runner(session, inp.name, x) if device in ("cuda", "trt") else None
        if bound is not None:
            row["bound_p50"] = round(sorted(_ms(bound) for _ in range(runs))[runs // 2], 2)
        out["batch"][n] = row
    if len(sizes) > 1:
        switches = []
        for _ in range(max(1, runs // len(sizes))):
            for n in sizes:  # every run here follows a run of another size
                switches.append(_ms(lambda n=n: session.run(None, {inp.name: feeds[n]})) - out["batch"][n]["p50"])
        out["switch_extra_ms"] = round(sorted(switches)[len(switches) // 2], 2)
    return out


def _first_input(path: Path) -> tuple[str, list]:
    """Name and shape of the model's first input, read without loading the weights."""
    import onnx

    inp = onnx.load(str(path), load_external_data=False).graph.input[0]
    return inp.name, [d.dim_value if d.HasField("dim_value") else d.dim_param for d in inp.type.tensor_type.shape.dim]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sizes = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8, 16]
    print("ONNX_SPEED " + json.dumps(measure(sys.argv[1], sizes, int(sys.argv[3]) if len(sys.argv) > 3 else 30)), flush=True)
