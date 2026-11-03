"""Eye crops: cut from the analysis frame (default) or from a higher-resolution copy of it."""

from __future__ import annotations

import cv2
import numpy as np

from streamsafe.analysis.analyzer import CROP_BACKGROUND, eye_crop


def _scene(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    img = rng.integers(100, 200, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), bool)
    mask[h // 4 : 3 * h // 4, w // 3 : w // 2] = True
    return img, mask


def test_full_resolution_crop_matches_the_box_and_keeps_the_person() -> None:
    small, mask = _scene(96, 54)
    big = cv2.resize(small, (192, 108), interpolation=cv2.INTER_NEAREST)
    box = (20, 5, 60, 50)
    low, high = eye_crop(small, box, mask), eye_crop(big, box, mask, 2.0)
    assert low.shape[:2] == (45, 40) and high.shape[:2] == (90, 80)
    # inside the person the high-res crop is the same picture, twice as large
    assert np.array_equal(high[40:50, 20:30], big[50:60, 60:70])
    # far from the person the background is darkened in both
    assert high[0, 0].max() <= int(big[10, 40].max() * CROP_BACKGROUND) + 1
    assert low[0, 0].max() <= int(small[5, 20].max() * CROP_BACKGROUND) + 1


def test_box_on_the_frame_edge_stays_inside_the_image() -> None:
    small, mask = _scene(96, 54)
    big = cv2.resize(small, (288, 162))
    crop = eye_crop(big, (80, 40, 96, 54), mask, 3.0)
    assert crop.shape[:2] == (42, 48)
