"""PyTorch vs ONNX backend, model by model, on the people of a video (GPU pod).

    python -m bench.backend_equiv VIDEO ONNX_DIR HEAD_DIR [N_FRAMES]

Prints one JSON line per model (prefix EQUIV): detector boxes / masks, eye risk p and how many people change
side of the threshold, pose keypoints, scene probabilities. The bar is not "identical": the PyTorch eye runs
in bf16, so a few borderline people already flip between bf16 and fp32 (bench/eye_equiv.py).
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from streamsafe.config import Config
from streamsafe.models.onnx_backend import load_onnx_models
from streamsafe.models.torch_backend import load_torch_models
from streamsafe.types import ANALYSIS_SIZE


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main(video: str, onnx_dir: str, head_dir: str, n: int = 60) -> None:
    cfg = Config.from_env(head_dir=head_dir, onnx_dir=onnx_dir)  # EYE_FP16=1 compares the fp16 eye
    t, o = load_torch_models(cfg), load_onnx_models(cfg)
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det = dict(frames=0, count_mismatch=0, box_px=[], mask_iou=[])
    eye = dict(pt=[], po=[])
    pose = dict(kp_px=[])
    scene = dict(p_diff=[])
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(full, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        a, b = t.detector(rgb), o.detector(rgb)
        det["frames"] += 1
        if len(a) != len(b):
            det["count_mismatch"] += 1
        for ba, ma in a:  # each PyTorch person against the ONNX person that overlaps it most (order can differ)
            if not b:
                break
            bb, mb = max(b, key=lambda d: _iou(ba, d[0]))
            det["box_px"].append(float(np.abs(np.array(ba) - np.array(bb)).max()))
            det["mask_iou"].append(float((ma & mb).sum() / max(1, (ma | mb).sum())))
        boxes = [bx for bx, _ in b if bx[3] - bx[1] >= cfg.detector.min_h]
        if boxes:
            s = full.shape[1] / ANALYSIS_SIZE[0]
            crops = [full[int(y0 * s) : int(y1 * s), int(x0 * s) : int(x1 * s)] for x0, y0, x1, y1 in boxes]
            crops = [c for c in crops if c.size]
            eye["pt"] += t.eye.prisk(t.eye.embed(crops)).tolist()
            eye["po"] += o.eye.prisk(o.eye.embed(crops)).tolist()
            for (ka, _), (kb, _) in zip(t.pose(rgb, boxes), o.pose(rgb, boxes), strict=True):
                pose["kp_px"].append(float(np.abs(ka - kb).max()))
        if t.scene is not None and o.scene is not None:
            _, pa = t.scene.embed_probs(rgb)
            _, pb = o.scene.embed_probs(rgb)
            scene["p_diff"].append(float(np.abs(pa - pb).max()))
    thr = t.eye.thr
    pt, po = np.array(eye["pt"]), np.array(eye["po"])

    def q(v: list[float]) -> dict:
        a = np.asarray(v)
        return dict(n=len(a), max=round(float(a.max()), 4), p95=round(float(np.percentile(a, 95)), 4)) if len(a) else {}

    covered_t, covered_o = pt >= thr, po >= thr
    out = {
        "detector": dict(
            frames=det["frames"],
            count_mismatch_frames=det["count_mismatch"],
            box_px=q(det["box_px"]),
            mask_iou_min=round(min(det["mask_iou"], default=1.0), 4),
        ),
        "eye": dict(
            people=len(pt),
            thr=thr,
            p_absdiff=q(np.abs(pt - po).tolist()),
            covered_torch=int(covered_t.sum()),
            covered_onnx=int(covered_o.sum()),
            only_torch=int((covered_t & ~covered_o).sum()),
            only_onnx=int((covered_o & ~covered_t).sum()),
        ),
        "pose": dict(kp_px=q(pose["kp_px"])),
        "scene": dict(p_absdiff=q(scene["p_diff"])),
    }
    for name, value in out.items():
        print("EQUIV", name, json.dumps(value), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(a) for a in sys.argv[4:5]))
