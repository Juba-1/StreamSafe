"""RF-DETR-Seg person detector on ONNX Runtime.

The network is the official RF-DETR ONNX export (``rfdetr`` ``export()``); pre- and post-processing here
reproduce ``RFDETR.predict`` + ``PostProcess`` exactly (rfdetr 1.11): bilinear resize without antialias to the
model resolution, ImageNet normalisation, sigmoid scores, top-``NUM_SELECT`` (query, class) pairs, boxes
cxcywh -> xyxy scaled to the frame and clamped, masks = bilinear upsample of the mask logits > 0.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from streamsafe.models.onnx_backend.session import Trt, open_session
from streamsafe.types import Box

COCO_PERSON = 1
NUM_SELECT = 100  # PostProcess.num_select of the published Seg checkpoints
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class OnnxRFDETR:
    def __init__(self, path: str | Path, thr: float = 0.35, allow_cpu: bool = False, trt: Trt | None = None) -> None:
        self.session = open_session(path, allow_cpu, trt)  # fixed input shape: no batch range needed
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.res = int(inp.shape[2])
        self.thr = thr

    def __call__(self, rgb: np.ndarray) -> list[tuple[Box, np.ndarray]]:
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb.astype(np.float32) / 255.0, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        boxes, logits, masks = self.session.run(None, {self.input_name: np.ascontiguousarray(x, np.float32)})
        return _postprocess(boxes[0], logits[0], masks[0], w, h, self.thr)


def _postprocess(boxes: np.ndarray, logits: np.ndarray, masks: np.ndarray, w: int, h: int, thr: float):
    q, c = logits.shape
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float32).reshape(-1)))
    order = np.argsort(-prob, kind="stable")[: min(NUM_SELECT, q * c)]
    scores, query, label = prob[order], order // c, order % c
    keep = scores > thr
    query, label = query[keep], label[keep]
    # the published weights use 1-based COCO ids; single-class fine-tunes use 0 (same rule as the torch backend)
    want = COCO_PERSON if (label == COCO_PERSON).any() else 0
    query = query[label == want]
    cx, cy, bw, bh = boxes[query].T
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) * np.array([w, h, w, h], np.float32)
    xyxy = np.minimum(np.maximum(xyxy, 0.0), np.array([w, h, w, h], np.float32))
    out = []
    for qi, b in zip(query, xyxy, strict=True):
        m = cv2.resize(masks[qi].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.0
        out.append((tuple(int(v) for v in b), m))
    return out
