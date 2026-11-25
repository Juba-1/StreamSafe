"""Where the start of the protection goes: open every model, then the warm-up (every batch size), each step
timed. One configuration per process, so nothing is cached in memory between them.
Usage: python tools/load_bench.py LABEL   (the runtime comes from the environment: TRT, TRT_BUILD, CUDNN_SEARCH ...)"""

import json
import sys
import time

import numpy as np


def main() -> None:
    label = sys.argv[1]
    t0 = time.perf_counter()
    import onnxruntime  # noqa: F401

    from streamsafe.config import Config
    from streamsafe.models import build_models
    from streamsafe.models.onnx_backend import session
    from streamsafe.types import ANALYSIS_SIZE

    t_import = time.perf_counter() - t0
    cfg = Config.from_env(mode="realtime")
    t1 = time.perf_counter()
    m = build_models(cfg)
    t_open = time.perf_counter() - t1
    w, h = ANALYSIS_SIZE
    blank = np.zeros((h, w, 3), np.uint8)
    steps = {}

    def timed(name, fn):
        a = time.perf_counter()
        fn()
        steps[name] = round(time.perf_counter() - a, 2)

    timed("detector", lambda: m.detector(blank))
    for n in range(1, 17):
        timed(f"eye{n}", lambda n=n: m.eye.embed([blank[:120, :60]] * n))
    for n in range(1, 9):
        timed(f"pose{n}", lambda n=n: m.pose(blank, [(10, 10, 60, 120)] * n))
    if m.scene is not None:
        timed("scene", lambda: m.scene(blank[..., ::-1].copy()))
    if m.hud is not None:
        timed("hud", lambda: m.hud(blank))
    # a second pass: what a frame costs once warm (the first pass minus this = one-time setup)
    again = {}
    for n in (1, 8, 16):
        a = time.perf_counter()
        m.eye.embed([blank[:120, :60]] * n)
        again[f"eye{n}"] = round(time.perf_counter() - a, 3)
    out = {
        "label": label,
        "devices": session.device_summary(),
        "notes": session.notes(),
        "import_s": round(t_import, 2),
        "open_models_s": round(t_open, 2),
        "warmup_s": round(sum(steps.values()), 2),
        "eye_warm_s": round(sum(v for k, v in steps.items() if k.startswith("eye")), 2),
        "pose_warm_s": round(sum(v for k, v in steps.items() if k.startswith("pose")), 2),
        "steps": steps,
        "warm_again": again,
    }
    print("LOADBENCH " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
