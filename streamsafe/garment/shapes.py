"""Where the garment goes on a person (pure geometry, translation invariant).

``cover_shape`` returns the region to cover and the region deliberately left visible (the head):
  * dress  - pose-shaped dress from the neckline to the knees when the skeleton is trustworthy;
             it always contains every body pixel of the torso / hip / thigh band;
  * robe   - loose robe along the body's main axis (PCA), head left visible when the pose allows;
  * no trustworthy pose at all -> the robe covers the whole person.
Safety: the region always contains the person's mask in the covered band; only the head is kept.
"""

from __future__ import annotations

from itertools import pairwise

import cv2
import numpy as np

KP_MIN = 0.6  # keypoint confidence needed for the shoulders / head
KP_DRESS = 0.5  # ...for hips, knees and arms

# COCO-17 keypoint indices
NOSE_TO_EARS = range(5)
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE = 11, 12, 13, 14

MIN_REGION = 30


def cover_shape(mask: np.ndarray, kp: np.ndarray | None = None, sc: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(cover, keep) for one person: the dress when the pose allows, else the robe."""
    try:
        region = dress_region(mask, kp, sc) if kp is not None else None
    except Exception:  # degenerate pose geometry: the robe is always safe
        region = None
    if region is None or region.sum() < MIN_REGION:
        return robe_cover(mask, kp, sc)
    return region, np.zeros_like(mask)


def robe_cover(mask: np.ndarray, kp: np.ndarray | None = None, sc: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(cover, keep): robe over the body, head visible when the pose is trustworthy; else full cover."""
    region, _ = robe_region(mask)
    keep = np.zeros_like(mask)
    if kp is None:
        return region, keep
    head = head_keep(mask, kp, sc)
    if head is None:
        return region, keep
    keep, above = head
    region = region & ~(above & ~mask)  # clean shoulder line: no fabric floating around the head
    shoulder_w = float(np.linalg.norm(kp[L_SHOULDER] - kp[R_SHOULDER]))
    ks = max(5, int(0.22 * shoulder_w)) | 1
    upper = (region & above & ~keep).astype(np.uint8)  # judge thickness WITHOUT the head window
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    thick = cv2.morphologyEx(upper, cv2.MORPH_OPEN, kernel).astype(bool)
    region = (region & ~above) | thick  # above the neckline keep only thick parts (raised arms)
    return region, keep


def robe_region(mask: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """Loose robe silhouette in the body's own axis, widened and flared downward; contains ``mask``."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return mask.copy(), None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, evec = np.linalg.eigh(np.cov((pts - c).T))
    axis = evec[:, 1]  # main body axis
    if axis[1] < 0:
        axis = -axis  # head -> feet is +
    angle = np.degrees(np.arctan2(axis[0], axis[1]))

    h, w = mask.shape
    rot = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), -angle, 1.0)
    big = int(np.hypot(h, w))
    rot[0, 2] += (big - w) / 2
    rot[1, 2] += (big - h) / 2
    upright = cv2.warpAffine(mask.astype(np.uint8), rot, (big, big), flags=cv2.INTER_NEAREST)

    rows = np.where(upright.any(1))[0]
    top, bot = rows.min(), rows.max()
    left, right = np.full(big, -1.0), np.full(big, -1.0)
    for y in rows:
        cols = np.where(upright[y])[0]
        left[y], right[y] = cols.min(), cols.max()
    height = max(1, bot - top)
    idx = np.arange(top, bot + 1)
    lv, rv = left[idx], right[idx]
    ok = lv >= 0
    lv = np.interp(idx, idx[ok], lv[ok])
    rv = np.interp(idx, idx[ok], rv[ok])
    k = max(3, int(height * 0.08)) | 1
    lv = cv2.GaussianBlur(lv.reshape(-1, 1), (1, k), 0).ravel()
    rv = cv2.GaussianBlur(rv.reshape(-1, 1), (1, k), 0).ravel()

    t = (idx - top) / height
    flare = (0.06 + 0.22 * np.clip((t - 0.2) / 0.8, 0, 1)) * height * 0.35  # wider toward the hem
    min_width = np.where(t > 0.18, 0.32 * height, 0)  # loose garment below the shoulders
    extra = np.maximum(0, (min_width - (rv - lv)) / 2)
    poly = np.concatenate([np.stack([lv - flare - extra, idx], 1), np.stack([rv + flare + extra, idx], 1)[::-1]])
    robe = np.zeros_like(upright)
    cv2.fillPoly(robe, [poly.astype(np.int32)], 1)
    ks = max(3, int(0.10 * height)) | 1
    robe = cv2.morphologyEx(robe, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    inv = cv2.invertAffineTransform(rot)
    region = cv2.warpAffine(robe, inv, (w, h), flags=cv2.INTER_NEAREST).astype(bool) | mask
    return region, {"R": rot, "Ri": inv, "big": big, "top": top, "bot": bot, "ang": angle}


def head_keep(mask: np.ndarray, kp: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(head region to leave visible, half-plane above the neckline), or None when the pose is not trustworthy."""
    h, w = mask.shape
    ls, rs = kp[L_SHOULDER], kp[R_SHOULDER]
    if sc[L_SHOULDER] < KP_MIN or sc[R_SHOULDER] < KP_MIN:
        return None
    sw = float(np.linalg.norm(ls - rs))
    if sw < 8:
        return None  # too small / side view collapsed
    head_idx = [i for i in NOSE_TO_EARS if sc[i] >= KP_MIN]
    if not head_idx:
        return None
    head_c = kp[head_idx].mean(0)
    mid = (ls + rs) / 2
    v = head_c - mid
    dist = float(np.linalg.norm(v))
    if dist < 0.25 * sw or dist > 2.5 * sw:
        return None  # head not where a head should be
    up = v / dist  # body "up": shoulders -> head
    if sc[L_HIP] >= KP_MIN or sc[R_HIP] >= KP_MIN:  # hips must be on the other side
        hips = kp[[i for i in (L_HIP, R_HIP) if sc[i] >= KP_MIN]].mean(0)
        if float(np.dot(hips - mid, up)) > 0:
            return None
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    along = (xs - mid[0]) * up[0] + (ys - mid[1]) * up[1]
    above = along > 0.18 * sw  # high neckline: covering starts a bit ABOVE the shoulders
    radius = max(0.55 * sw, 1.05 * dist)
    disc = (xs - head_c[0]) ** 2 + (ys - head_c[1]) ** 2 <= radius**2
    keep = above & disc & mask
    if keep.sum() < 20:
        return None
    return keep, above


def dress_region(mask: np.ndarray, kp: np.ndarray, sc: np.ndarray, oversize: float = 1.12) -> np.ndarray | None:
    """Pose-shaped dress region, or None when the pose is not trustworthy (the robe is used instead).

    Region = smooth dress shape  U  every person pixel in the neckline->hem band that is not an
    outer arm, so torso, chest, hips and thighs are covered even if the shape is too narrow.
    """
    h, w = mask.shape
    if kp is None or sc[L_SHOULDER] < KP_MIN or sc[R_SHOULDER] < KP_MIN:
        return None
    hip_i = [i for i in (L_HIP, R_HIP) if sc[i] >= KP_DRESS]
    ls, rs = kp[L_SHOULDER].astype(np.float32), kp[R_SHOULDER].astype(np.float32)
    sw = float(np.linalg.norm(ls - rs))
    sh = (ls + rs) / 2
    if hip_i:
        hp = kp[hip_i].mean(0).astype(np.float32)
    else:
        # close-up cut by the bottom of the frame: top-style garment from the shoulders to the edge
        if not mask.any() or np.nonzero(mask)[0].max() < h - 3 or sw < 20:
            return None
        v = (rs - ls) / max(sw, 1e-6)
        dn = np.array([-v[1], v[0]], np.float32)
        if dn[1] < 0:
            dn = -dn
        hp = sh + dn * 1.3 * sw
    down = hp - sh
    torso = float(np.linalg.norm(down))
    if torso < 10 or sw < 6:
        return None
    down /= torso
    if sw > 2.2 * torso:
        return None  # lying / foreshortened: geometry unreliable
    if down[1] < 0.6:
        return None  # not upright (lying, bent, rolled camera) -> robe
    perp = np.array([-down[1], down[0]], np.float32)
    knees = [i for i in (L_KNEE, R_KNEE) if sc[i] >= KP_DRESS]
    hem = max(float(np.dot(kp[i] - sh, down)) for i in knees) if knees else 1e9  # knees unseen: to the body's end
    hem = max(hem, 1.35 * torso)  # never shorter than mid-thigh

    ys0, xs0 = np.nonzero(mask)
    if len(xs0) < 50:
        return None
    pad = int(0.5 * sw) + 6  # work inside the person's box only (speed)
    x1, y1 = max(0, xs0.min() - pad), max(0, ys0.min() - pad)
    x2, y2 = min(w, xs0.max() + pad + 1), min(h, ys0.max() + pad + 1)
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    a = (xx - sh[0]) * down[0] + (yy - sh[1]) * down[1]  # along the body (0 = shoulders, torso = hips)
    q = (xx - sh[0]) * perp[0] + (yy - sh[1]) * perp[1]  # across the body
    centre0 = float(np.dot((ls + rs) / 2 - sh, perp))  # 0
    top = 0.06 * torso  # neckline just below the shoulder joints
    hip_w = max(float(np.linalg.norm(kp[L_HIP] - kp[R_HIP])) if len(hip_i) == 2 else 0.0, 0.6 * sw if hip_i else 0.75 * sw)

    def half_width(t: np.ndarray) -> np.ndarray:
        """Oversized profile: bust ~ shoulders, waist a bit in, hips, then A-line flare to the hem."""
        t = np.asarray(t, np.float32)
        bust, waist = 0.50 * sw, 0.46 * sw
        hips = max(0.55 * sw, 0.62 * hip_w)
        stops = [top, 0.45 * torso, 0.75 * torso, 1.05 * torso, max(hem, 1.06 * torso)]
        return oversize * np.interp(t, stops, [bust * 0.92, bust, waist, hips, hips * 1.18])

    band = (a >= top) & (a <= hem)
    # the body axis can drift with the pelvis: centre the skirt on the hips below the waist
    hip_c = float(np.dot(hp - sh, perp))
    centre = np.where(a < 0.6 * torso, centre0, centre0 + (hip_c - centre0) * np.clip((a - 0.6 * torso) / (0.4 * torso), 0, 1))
    shape = band & (np.abs(q - centre) <= half_width(a))

    # arms: capsules shoulder->elbow->wrist; an arm over the dress stays visible only OUTSIDE the shape
    m = mask[y1:y2, x1:x2]
    arm8 = np.zeros(m.shape, np.uint8)
    radius = max(3, int(0.17 * sw))
    for s_, e_, w_ in ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)):
        pts = [kp[i] for i in (s_, e_, w_) if sc[i] >= KP_DRESS]
        if sc[s_] >= KP_MIN and len(pts) >= 2:
            for p0, p1 in pairwise(pts):
                cv2.line(arm8, (int(p0[0] - x1), int(p0[1] - y1)), (int(p1[0] - x1), int(p1[1] - y1)), 1, 2 * radius)
    arm = arm8.astype(bool)
    reach = np.maximum(1.6 * half_width(a), 0.5 * torso)  # torso / hip / thigh pixels are within this of the axis
    need = m & band & ~(arm & ~shape) & (np.abs(q - centre) <= reach)
    region = shape | need

    # tidy: fill across-body gaps row by row (skirt between the legs), then smooth the outline
    rows = np.clip(np.round(a - top).astype(np.int32), 0, None)
    n_rows = int(min(hem, a.max()) - top) + 2
    if n_rows > 2 and region.any():
        lo = np.full(n_rows, 1e9, np.float32)
        hi = np.full(n_rows, -1e9, np.float32)
        np.minimum.at(lo, rows[region], q[region])
        np.maximum.at(hi, rows[region], q[region])
        ok = hi > lo
        lo_s, hi_s = np.where(ok, lo, np.nan), np.where(ok, hi, np.nan)
        r = np.clip(rows, 0, n_rows - 1)
        fill = band & (q >= lo_s[r] - 1) & (q <= hi_s[r] + 1)
        fill &= ~(arm & ~shape & ~m)  # no sleeve painted into the gap beside an arm
        region = region | np.nan_to_num(fill, nan=0).astype(bool)
    smooth = cv2.GaussianBlur(region.astype(np.float32), (0, 0), max(1.0, 0.03 * sw)) > 0.5
    region = smooth | need  # smoothing never removes a required pixel
    out = np.zeros((h, w), bool)
    out[y1:y2, x1:x2] = region
    return out
