"""What posture costs at run time: ViTPose on 1 / 4 / 8 people of a real frame, plus the posture features,
next to the eye on the same people.

usage: python -m experiments.pose_speed HEAD_DIR VIDEO
Prints "POSE_SPEED <json>" (median ms over 30 runs after warm-up).
"""

from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.analysis.posture import features
from streamsafe.config import Config
from streamsafe.io.sources import VideoFileSource
from streamsafe.models import build_models
from streamsafe.types import ANALYSIS_SIZE

RUNS = 30


def _ms(fn) -> float:
    for _ in range(5):
        fn()
    t = []
    for _ in range(RUNS):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return round(float(np.median(t)), 2)


def main(argv: list[str]) -> None:
    cfg = Config.from_env(head_dir=argv[0], scene_prior=False)
    models = build_models(cfg)
    analyzer = Analyzer(cfg, models, output_scale=1.0)
    source = VideoFileSource(argv[1], 2, 60, size=ANALYSIS_SIZE)
    best = None
    for frame in source.frames():  # the frame with the most people
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        people, crops = analyzer.detect(rgb)
        if best is None or len(people) > len(best[1]):
            best = (rgb, people, crops)
    source.close()
    rgb, people, crops = best
    rep: dict = {"people_in_frame": len(people)}
    for n in (1, 4, 8):
        k = min(n, len(people))
        boxes = [p.box for p in people[:k]] * (n // k + 1)
        boxes, cr = boxes[:n], (crops[:k] * (n // k + 1))[:n]
        kps = models.pose(rgb, boxes)
        rep[f"pose_{n}"] = _ms(lambda b=boxes: models.pose(rgb, b))
        rep[f"eye_{n}"] = _ms(lambda c=cr: models.eye.embed(c))
        rep[f"features_{n}"] = _ms(
            lambda q=kps, b=boxes: [features(kp, ks, (x[2] - x[0]) / max(1, x[3] - x[1])) for (kp, ks), x in zip(q, b, strict=True)]
        )
    print("POSE_SPEED " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
