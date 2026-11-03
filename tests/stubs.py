"""Stand-in models for CPU tests (no torch). People are solid ellipses:
RED = risky woman, GREEN = clearly safe person, BLUE = borderline safe (p 0.06).
Detector = colour blobs; Eye = colour of the crop; Pose = no keypoints (garment falls back to the robe)."""

from __future__ import annotations

import cv2
import numpy as np

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.config import Config
from streamsafe.models import Models
from streamsafe.types import Box

RED, GREEN, BLUE = (40, 40, 220), (40, 200, 40), (220, 60, 40)  # BGR
MIN_BLOB = 150


class BlobDetector:
    def __call__(self, rgb: np.ndarray) -> list[tuple[Box, np.ndarray]]:
        out = []
        for col in (RED, GREEN, BLUE):
            target = np.array(col[::-1], np.int16)
            hit = (np.abs(rgb.astype(np.int16) - target).sum(2) < 60).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(hit)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area > MIN_BLOB:
                    out.append(((int(x), int(y), int(x + w), int(y + h)), labels == i))
        return out


class ColourEye:
    """Identity dims are equal for red and blue (same person changing outfit), different for green."""

    thr = 0.1746

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = []
        for c in crops:
            px = c.reshape(-1, 3).astype(np.float32)
            lit = px.sum(1) > 60
            mean = px[lit].mean(0) if lit.any() else np.zeros(3)
            red = mean[0] > max(mean[1], mean[2])
            green = mean[1] > max(mean[0], mean[2])
            blue = mean[2] > max(mean[0], mean[1])
            e = np.zeros(16, np.float32)
            e[0] = 0.3 if red else -0.3
            e[1] = 0.3 if blue else 0.0
            e[4:10] = 0.0 if green else 1.0
            e[10:16] = 1.0 if green else 0.0
            out.append(e / np.linalg.norm(e))
        return np.array(out)

    def prisk(self, emb: np.ndarray) -> np.ndarray:
        return np.where(emb[:, 0] > 0, 0.95, np.where(emb[:, 1] > 0.05, 0.06, 0.02))


class NoPose:
    def __call__(self, rgb: np.ndarray, boxes: list[Box]) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(np.zeros((17, 2), np.float32), np.zeros(17, np.float32)) for _ in boxes]


def stub_analyzer(cfg: Config, output_scale: float) -> Analyzer:
    """Analyzer factory for the tests (top-level so the realtime worker process can import it)."""
    return Analyzer(cfg, Models(detector=BlobDetector(), eye=ColourEye(), pose=NoPose()), output_scale)


def make_video(
    path: str,
    n: int = 120,
    cut_at: int = 60,
    speed: int = 6,
    turn_safe_at: int | None = None,
    turn_col: tuple[int, int, int] = GREEN,
    texture: bool = True,
    flash_at: int | None = None,
    size: tuple[int, int] = (1920, 1080),
    fold_shade: float = 0.8,
    first_col: tuple[int, int, int] = RED,
) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Red person walks right (``speed`` px/frame at 1080p); green person static; new background
    at ``cut_at``. ``fold_shade`` 0.8 splits the stub blob into pieces (as in engine2's tests), 0.9
    keeps it whole. ``first_col`` / ``turn_col`` / ``turn_safe_at``: the walker's outfit before / after
    a change. Returns per frame the ground-truth masks (analysis size) of red and green."""
    w, h = size
    k = w / 1920
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    truth = []
    for f in range(n):
        bg = (90, 120, 140) if f < cut_at else (150, 90, 60)
        im = np.full((h, w, 3), bg, np.uint8)
        cv2.rectangle(im, (0, int(900 * k)), (w, h), (60, 60, 60), -1)
        cx = 400 + speed * f
        col = turn_col if (turn_safe_at is not None and f >= turn_safe_at) else first_col
        cv2.ellipse(im, (int(cx * k), int(600 * k)), (int(70 * k), int(230 * k)), 0, 0, 360, col, -1)
        if texture:  # clothing folds, so motion compensation has something to follow
            shade = tuple(int(v * fold_shade) for v in col)
            for j in range(8):
                p0, p1 = (int((cx - 60) * k), int((450 + 40 * j) * k)), (int((cx + 60) * k), int((470 + 40 * j) * k))
                cv2.line(im, p0, p1, shade, max(1, int(6 * k)))
        cv2.ellipse(im, (int(1500 * k), int(600 * k)), (int(70 * k), int(230 * k)), 0, 0, 360, GREEN, -1)
        red = np.zeros((540, 960), np.uint8)
        cv2.ellipse(red, (cx // 2, 300), (35, 115), 0, 0, 360, 1, -1)
        green = np.zeros((540, 960), np.uint8)
        cv2.ellipse(green, (750, 300), (35, 115), 0, 0, 360, 1, -1)
        if flash_at is not None and f == flash_at:
            im[:] = 245  # one-frame white flash: a cut in and a cut out
        truth.append((red.astype(bool), green.astype(bool), col == RED))
        writer.write(im)
    writer.release()
    return truth
