"""Fixed rules on top of the eye for women the eye reads as modest (Ali, 14 Nov 2026):

  * BACK: a big woman seen from behind (face not visible, or her left shoulder and hip on the image's left)
    is covered: the game's camera dwells on the back view even with modest clothes.
  * HIPS: a close-up of the hips with no head or shoulders in the picture is covered.
  * MIDRIFF: a bare belly (a top that shows the stomach = "revealing" in the policy) is covered. Skin is
    judged against the woman's own face, so it works for every skin tone and lighting.

They only ever add a cover (a garment, never the whole screen) and never remove one. Pure functions on COCO-17
keypoints so they are testable without models."""

from __future__ import annotations

import cv2
import numpy as np

NOSE, LEYE, REYE = 0, 1, 2
LSHO, RSHO, LHIP, RHIP = 5, 6, 11, 12
SEEN = 0.4  # keypoint score to count as visible
FACE_HIDDEN = 0.3
BACK_MIN_H = 0.45  # x frame height: a back view this tall is covered
BACK_MIN_HIP_W = 0.10  # x frame width: ...or hips this wide
HIPS_CLOSE_W = 0.15  # x frame width: hips this wide with no head / shoulders in the picture
MIDRIFF_SKIN = 0.40  # share of the belly band that matches the face skin


def back_view(kp: np.ndarray, sc: np.ndarray) -> bool:
    body = min(sc[LSHO], sc[RSHO], sc[LHIP], sc[RHIP]) >= SEEN
    if not body:
        return False
    face = max(sc[NOSE], sc[LEYE], sc[REYE])
    turned = kp[LSHO, 0] < kp[RSHO, 0] and kp[LHIP, 0] < kp[RHIP, 0]  # her left side on the image's left
    return face < FACE_HIDDEN or (turned and face < 0.6)


def hips_close(kp: np.ndarray, sc: np.ndarray, size: tuple[int, int]) -> bool:
    w = size[0]
    hips = min(sc[LHIP], sc[RHIP]) >= SEEN
    no_top = max(sc[NOSE], sc[LEYE], sc[REYE], sc[LSHO], sc[RSHO]) < FACE_HIDDEN
    return hips and no_top and abs(kp[LHIP, 0] - kp[RHIP, 0]) >= HIPS_CLOSE_W * w


def big_back(kp: np.ndarray, sc: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> bool:
    w, h = size
    tall = (box[3] - box[1]) >= BACK_MIN_H * h
    wide = abs(kp[LHIP, 0] - kp[RHIP, 0]) >= BACK_MIN_HIP_W * w
    return back_view(kp, sc) and (tall or wide)


def bare_midriff(bgr: np.ndarray, mask: np.ndarray, kp: np.ndarray, sc: np.ndarray) -> bool:
    if min(sc[NOSE], sc[LSHO], sc[RSHO], sc[LHIP], sc[RHIP]) < SEEN:
        return False
    sho_y, hip_y = (kp[LSHO, 1] + kp[RSHO, 1]) / 2, (kp[LHIP, 1] + kp[RHIP, 1]) / 2
    torso = hip_y - sho_y
    if torso < 20:  # too small to judge
        return False
    hh, ww = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = max(2, int(0.08 * torso))
    nx, ny = int(kp[NOSE, 0]), int(kp[NOSE, 1])
    face = lab[max(0, ny - r) : min(hh, ny + r), max(0, nx - r) : min(ww, nx + r)].reshape(-1, 3)
    if len(face) < 9:
        return False
    skin = np.median(face, axis=0)
    y0, y1 = int(sho_y + 0.55 * torso), int(hip_y)
    xs = [kp[LSHO, 0], kp[RSHO, 0], kp[LHIP, 0], kp[RHIP, 0]]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1, x0, x1 = max(0, y0), min(hh, y1), max(0, x0), min(ww, x1)
    if y1 - y0 < 4 or x1 - x0 < 4:
        return False
    band = lab[y0:y1, x0:x1]
    inside = mask[y0:y1, x0:x1]
    if inside.sum() < 30:
        return False
    d_ab = np.linalg.norm(band[..., 1:] - skin[1:], axis=-1)
    d_l = np.abs(band[..., 0] - skin[0])
    match = (d_ab < 14) & (d_l < 40) & inside
    return float(match.sum()) / float(inside.sum()) >= MIDRIFF_SKIN


def hit(bgr: np.ndarray, mask: np.ndarray, box, kp: np.ndarray, sc: np.ndarray) -> str:
    """The rule a woman's picture breaks ('back' / 'hips' / 'midriff') or ''."""
    size = (bgr.shape[1], bgr.shape[0])
    if big_back(kp, sc, box, size):
        return "back"
    if hips_close(kp, sc, size):
        return "hips"
    if bare_midriff(bgr, mask, kp, sc):
        return "midriff"
    return ""
