"""Posture features from COCO-17 keypoints: how a person is placed, independent of the clothes.

The eye judges the outfit; these numbers describe the pose (lying, bent over, crouching, legs apart, seen from
behind, arms up). They are cheap (pure numpy on keypoints the pose model already gives) and are used to judge
women more strictly when the pose is suggestive. Keypoints are in box units: x, y in 0..1 from the box's
top-left corner, so the features do not depend on the person's size.
"""

from __future__ import annotations

import numpy as np

NOSE, EYES, SHOULDERS, WRISTS, HIPS, KNEES, ANKLES = 0, (1, 2), (5, 6), (9, 10), (11, 12), (13, 14), (15, 16)
SEEN = 0.3  # keypoint score above which a joint counts as visible
NAMES = (
    "aspect", "tilt", "head_below_hips", "hip_y", "knee_spread", "ankle_spread", "knee_bend",
    "hip_shoulder_ratio", "face_seen", "back_view", "arms_up", "seen_share",
)  # fmt: skip


def _mid(kp: np.ndarray, ks: np.ndarray, pair: tuple[int, int]) -> np.ndarray | None:
    ok = [i for i in pair if ks[i] >= SEEN]
    return kp[ok].mean(0) if ok else None


def _width(kp: np.ndarray, ks: np.ndarray, pair: tuple[int, int]) -> float | None:
    a, b = pair
    return float(abs(kp[a, 0] - kp[b, 0])) if ks[a] >= SEEN and ks[b] >= SEEN else None


def _bend(kp: np.ndarray, ks: np.ndarray, hip: int, knee: int, ankle: int) -> float | None:
    """0 = straight leg, 1 = folded flat."""
    if min(ks[hip], ks[knee], ks[ankle]) < SEEN:
        return None
    u, v = kp[hip] - kp[knee], kp[ankle] - kp[knee]
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))
    return 1.0 - float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))) / 180.0


def features(kp: np.ndarray, ks: np.ndarray, aspect: float) -> np.ndarray:
    """(17, 2) keypoints in box units + (17,) scores + box width / height -> posture vector (len(NAMES),).
    A joint that is not visible gives a neutral value (0), never an error."""
    sh, hip = _mid(kp, ks, SHOULDERS), _mid(kp, ks, HIPS)
    tilt = 0.0
    if sh is not None and hip is not None:
        d = hip - sh
        tilt = float(abs(np.degrees(np.arctan2(d[0], d[1] + 1e-6)))) / 90.0  # 0 upright, 1 horizontal, 2 upside down
    head_below = float(ks[NOSE] >= SEEN and hip is not None and kp[NOSE, 1] > hip[1])
    shoulder_w = _width(kp, ks, SHOULDERS) or 0.0
    norm = max(shoulder_w, 0.15)

    def spread(pair: tuple[int, int]) -> float:
        w = _width(kp, ks, pair)
        return min(w / norm, 4.0) if w is not None else 0.0

    bends = [b for b in (_bend(kp, ks, 11, 13, 15), _bend(kp, ks, 12, 14, 16)) if b is not None]
    hip_w = _width(kp, ks, HIPS)
    face = float(np.mean(ks[[NOSE, *EYES]]))
    shoulders_seen = float(np.mean(ks[list(SHOULDERS)]))
    wrist_y = [kp[i, 1] for i in WRISTS if ks[i] >= SEEN]
    arms_up = float(ks[NOSE] >= SEEN and bool(wrist_y) and min(wrist_y) < kp[NOSE, 1])
    return np.array([
        min(aspect, 3.0), tilt, head_below, float(hip[1]) if hip is not None else 0.5, spread(KNEES), spread(ANKLES),
        float(np.mean(bends)) if bends else 0.0, min(hip_w / norm, 3.0) if hip_w is not None and shoulder_w > 0 else 0.0,
        face, float(face < SEEN and shoulders_seen >= SEEN), arms_up, float(np.mean(ks >= SEEN)),
    ], np.float32)  # fmt: skip
