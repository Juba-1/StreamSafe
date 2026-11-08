"""Turn a moment of a video into labeled training examples for the eye's heads (e.g. Ali: "Lucia in the
prison scene is modest, she must not be covered").

The engine's own detector and eye (full-resolution crops, exactly as live) run on every frame between T0 and
T1; the biggest person of each frame becomes one example with the given label (R revealing, S modest woman,
M man). A contact sheet of the chosen crops is written next to the output, so a person can check that the
right character was picked before the examples are used.

usage: HEAD_DIR=<heads> python -m training.add_examples OUT.npz VIDEO T0 T1 LABEL [FAMILY]
Writes OUT.npz (X fp16 unit-norm, C label, G family) and OUT.jpg; prints "EXAMPLES <json>".
Use the npz with ``EXTRA=OUT.npz python -m training.compare_heads ...`` (training only, never a test fold).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.config import Config
from streamsafe.io.sources import VideoFileSource
from streamsafe.models import build_models
from streamsafe.types import ANALYSIS_SIZE

FPS = 10.0
THUMB_H = 160


def main(argv: list[str]) -> None:
    out, video, t0, t1, label = Path(argv[0]), argv[1], float(argv[2]), float(argv[3]), argv[4].upper()
    family = argv[5] if len(argv) > 5 else "extra"
    if label not in ("R", "S", "M"):
        raise SystemExit("LABEL must be R, S or M")
    cfg = Config.from_env()
    analyzer = Analyzer(cfg, build_models(cfg), 1.0)
    src = VideoFileSource(video, FPS, t1)
    x, thumbs, ps = [], [], []
    for k, frame in enumerate(src.frames()):
        if k / FPS < t0:
            continue
        rgb = cv2.cvtColor(cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        people, crops = analyzer.detect(rgb, frame, full_bgr=True)
        if not people:
            continue
        j = max(range(len(people)), key=lambda i: (people[i].box[3] - people[i].box[1]) * (people[i].box[2] - people[i].box[0]))
        emb = analyzer.models.eye.embed([crops[j]])
        x.append(emb[0])
        ps.append(float(analyzer.models.eye.prisk(emb)[0]))
        c = crops[j]
        thumbs.append(cv2.resize(cv2.cvtColor(c, cv2.COLOR_RGB2BGR), (max(1, int(c.shape[1] * THUMB_H / c.shape[0])), THUMB_H)))
    src.close()
    if not x:
        raise SystemExit("no person found between T0 and T1")
    xs = np.stack(x).astype(np.float32)
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)
    np.savez_compressed(out, X=xs.astype(np.float16), C=np.array([label] * len(xs)), G=np.array([family] * len(xs)))
    cv2.imwrite(str(out.with_suffix(".jpg")), np.hstack(thumbs[:12]))
    rep = dict(n=len(xs), label=label, family=family, p_now=[round(p, 3) for p in ps], thr=round(analyzer.thr, 4))
    print("EXAMPLES " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
