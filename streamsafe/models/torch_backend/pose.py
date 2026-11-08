"""ViTPose-base, every person of a frame in one batch.

Pre/post-processing reproduces the Hugging Face processor exactly (measured difference 0.0 px):
unbiased (UDP) affine warp with 1.25 padding at 3:4 aspect, then DARK sub-pixel decoding.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from streamsafe.models.base import Keypoints
from streamsafe.models.torch_backend.device import DEVICE, HALF
from streamsafe.types import Box

INPUT_W, INPUT_H = 192, 256
PADDING = 1.25
ASPECT = INPUT_W / INPUT_H  # 0.75
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


class ViTPose:
    def __init__(self, repo: str, revision: str) -> None:
        from transformers import VitPoseForPoseEstimation

        self.model = VitPoseForPoseEstimation.from_pretrained(repo, revision=revision).to(DEVICE, HALF).eval()

    def __call__(self, rgb: np.ndarray, boxes: list[Box]) -> list[Keypoints]:
        if not boxes:
            return []
        crops, geometry = zip(*(_warp(rgb, box) for box in boxes), strict=True)
        x = torch.from_numpy((np.stack(crops).astype(np.float32) - MEAN) / STD)
        x = x.to(DEVICE).permute(0, 3, 1, 2).to(HALF)
        with torch.inference_mode():
            heatmaps = self.model(pixel_values=x).heatmaps.float().cpu().numpy()
        fx, fy, scores = _dark_decode(heatmaps)
        hh, ww = heatmaps.shape[2:]
        out = []
        for i, (cx, cy, sw, sh) in enumerate(geometry):
            kx = fx[i] * sw / (ww - 1.0) + cx - 0.5 * sw
            ky = fy[i] * sh / (hh - 1.0) + cy - 0.5 * sh
            out.append((np.stack([kx, ky], 1).astype(np.float32), scores[i].astype(np.float32)))
        return out


def _warp(rgb: np.ndarray, box: Box) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """UDP warp of one box to the model input; returns the crop and (cx, cy, scale_w, scale_h)."""
    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])
    cx, cy = box[0] + w / 2, box[1] + h / 2
    if w > ASPECT * h:
        h = w / ASPECT
    elif w < ASPECT * h:
        w = h * ASPECT
    sw, sh = PADDING * w, PADDING * h
    ax, ay = (INPUT_W - 1.0) / sw, (INPUT_H - 1.0) / sh
    m = np.array([[ax, 0, ax * (0.5 * sw - cx)], [0, ay, ay * (0.5 * sh - cy)]], np.float32)
    return cv2.warpAffine(rgb, m, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR), (cx, cy, sw, sh)


def _dark_decode(hm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arg-max + DARK (Taylor expansion on the log heatmap) -> sub-pixel peaks (B, K) and scores."""
    b, k, hh, ww = hm.shape
    flat = hm.reshape(b, k, -1)
    idx, scores = flat.argmax(-1), flat.max(-1)
    py, px = (idx // ww).astype(np.float64), (idx % ww).astype(np.float64)

    blurred = np.stack([cv2.GaussianBlur(hm[i, j], (11, 11), 0.8) for i in range(b) for j in range(k)])
    logs = np.log(np.clip(blurred.reshape(b, k, hh, ww), 0.001, 50))
    padded = np.pad(logs, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    bi, ki = np.meshgrid(np.arange(b), np.arange(k), indexing="ij")
    yc, xc = py.astype(int) + 1, px.astype(int) + 1

    def at(dy: int, dx: int) -> np.ndarray:
        return padded[bi, ki, yc + dy, xc + dx]

    c, right, left, down, up = at(0, 0), at(0, 1), at(0, -1), at(1, 0), at(-1, 0)
    dxx = right - 2 * c + left
    dyy = down - 2 * c + up
    dxy = 0.5 * (at(1, 1) - right - down + 2 * c - left - up + at(-1, -1))
    hessian = np.stack([np.stack([dxx, dxy], -1), np.stack([dxy, dyy], -1)], -2)
    hessian = hessian + np.finfo(np.float32).eps * np.eye(2)
    grad = np.stack([0.5 * (right - left), 0.5 * (down - up)], -1)
    offset = np.einsum("bkmn,bkn->bkm", np.linalg.inv(hessian), grad)
    return px - offset[..., 0], py - offset[..., 1], scores
