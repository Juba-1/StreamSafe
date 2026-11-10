"""How the garment looks: opaque cloth shaded by the scene's light, with folds and edge volume.

Nothing of the person shows through: the shading only uses the heavily blurred scene light
(no body detail), so the cloth is a real cover, never a filter.
"""

from __future__ import annotations

import cv2
import numpy as np

# BGR
PALETTE: dict[str, tuple[int, int, int]] = {
    "navy": (70, 42, 30),
    "black": (34, 30, 28),
    "sand": (150, 178, 196),
    "maroon": (45, 35, 110),
    "olive": (60, 95, 90),
    "grey": (110, 110, 110),
    "white": (225, 228, 230),
}
N_FOLDS = 6
EDGE_FEATHER = 0.12  # a logo without transparency fades in over this share of its size (no hard square edge)


def parse_color(value: str) -> tuple[int, int, int]:
    """BGR from a palette name or "#RRGGBB"."""
    v = value.strip()
    if v.lower() in PALETTE:
        return PALETTE[v.lower()]
    h = v.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"colour must be a palette name {sorted(PALETTE)} or #RRGGBB, got {value!r}")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def fabric(im: np.ndarray, region: np.ndarray, info: dict | None, color: tuple[int, int, int] | np.ndarray, seed: int = 0) -> np.ndarray:
    """Float BGR cloth for ``region`` (same size as ``im``). ``info`` = robe frame (folds follow the body axis).

    ``color``: a BGR triple, or an image texture tiled over the garment box.
    """
    h, w = region.shape
    ys, xs = np.nonzero(region)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    height = max(10, y2 - y1)

    lum = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    light = cv2.GaussianBlur(lum, (0, 0), max(12, 0.22 * height))  # scene light only, no body detail
    light = 0.80 + 0.45 * (light - light[region].mean())
    fold = _folds(h, w, height, info, seed) if info is not None else 0
    dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    volume = 0.62 + 0.38 * np.clip(dist / max(3.0, 0.06 * height), 0, 1) ** 0.6
    noise = cv2.GaussianBlur(np.random.RandomState(seed + 1).rand(h, w).astype(np.float32), (0, 0), 0.8) * 0.06 - 0.03
    shade = np.clip(light * volume * (1 + fold) + noise, 0.35, 1.35)

    if isinstance(color, np.ndarray) and color.ndim == 3:
        base = _tile(color, h, w, (y1, y2, x1, x2))
        out = base * shade[..., None]
    else:
        out = np.array(color, np.float32)[None, None] * shade[..., None]
    return np.clip(out, 0, 255)


def _folds(h: int, w: int, height: int, info: dict, seed: int) -> np.ndarray:
    """Soft vertical folds along the body axis, deeper toward the hem."""
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rot = info["R"]
    u = rot[0, 0] * gx + rot[0, 1] * gy + rot[0, 2]
    v = rot[1, 0] * gx + rot[1, 1] * gy + rot[1, 2]
    t = np.clip((v - info["top"]) / max(1, info["bot"] - info["top"]), 0, 1)
    rng = np.random.RandomState(seed)
    fold = np.zeros((h, w), np.float32)
    for _ in range(N_FOLDS):
        wavelength = rng.uniform(0.06, 0.16) * height
        amp = rng.uniform(0.010, 0.028)
        phase = rng.uniform(0, 2 * np.pi)
        drift = rng.uniform(-0.6, 0.6)  # folds lean / wander slightly down the body
        fold += amp * np.sin(2 * np.pi * (u + drift * (v - info["top"]) * 0.25) / wavelength + phase)
    fold = np.sign(fold) * np.abs(fold) ** 0.8  # softer crests
    return fold * (0.35 + 0.85 * t)


def _tile(texture: np.ndarray, h: int, w: int, box: tuple[int, int, int, int]) -> np.ndarray:
    y1, y2, x1, x2 = box
    tex = texture.astype(np.float32)
    th, tw = tex.shape[:2]
    s = max(0.2, 0.45 * (x2 - x1) / tw)
    tex = cv2.resize(tex, (max(4, int(tw * s)), max(4, int(th * s))))
    th, tw = tex.shape[:2]
    base = np.zeros((h, w, 3), np.float32)
    for yy in range(y1, y2, th):
        for xx in range(x1, x2, tw):
            hh, ww = min(th, h - yy), min(tw, w - xx)
            base[yy : yy + hh, xx : xx + ww] = tex[:hh, :ww]
    return base


def load_logo(path: str) -> np.ndarray | None:
    """BGRA logo image, or None when no / unreadable path."""
    if not path:
        return None
    try:  # np.fromfile + imdecode: cv2.imread cannot open non-ASCII paths on Windows (an Arabic user name)
        logo = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    except OSError:
        return None
    if logo is None:
        return None
    if logo.ndim == 2:
        logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGRA)
    if logo.shape[2] == 3:  # no transparency: opaque, with softly faded edges
        h, w = logo.shape[:2]
        f = max(1.0, EDGE_FEATHER * min(h, w))
        ramp_y = np.clip((np.minimum(np.arange(h), h - 1 - np.arange(h)) + 0.5) / f, 0, 1)
        ramp_x = np.clip((np.minimum(np.arange(w), w - 1 - np.arange(w)) + 0.5) / f, 0, 1)
        alpha = (255 * np.outer(ramp_y, ramp_x) ** 1.5).astype(np.uint8)  # the image itself stays exact
        logo = np.dstack([logo, alpha])
    return logo


def put_logo(fab: np.ndarray, region: np.ndarray, logo: np.ndarray) -> np.ndarray:
    """Logo on the chest band of the garment (inside the garment only), in its own exact colours."""
    ys, xs = np.nonzero(region)
    if len(ys) < 200:
        return fab
    top, bot = ys.min(), ys.max()
    height = bot - top
    band = (ys >= top + 0.08 * height) & (ys <= top + 0.45 * height)
    if band.sum() < 50:
        return fab
    cx = int(np.median(xs[band]))
    lw = int(0.85 * (np.percentile(xs[band], 90) - np.percentile(xs[band], 10)))
    if lw < 12:
        return fab
    lh = min(max(1, int(lw * logo.shape[0] / logo.shape[1])), int(0.42 * height))
    lw = max(1, int(lh * logo.shape[1] / logo.shape[0]))
    scaled = cv2.resize(logo, (lw, lh), interpolation=cv2.INTER_AREA)
    cy = int(top + 0.08 * height + lh // 2 + 0.02 * height)
    y0, x0 = cy - lh // 2, cx - lw // 2
    a1, b1 = max(0, y0), max(0, x0)
    a2, b2 = min(fab.shape[0], y0 + lh), min(fab.shape[1], x0 + lw)
    if a2 <= a1 or b2 <= b1:
        return fab
    g = scaled[a1 - y0 : a2 - y0, b1 - x0 : b2 - x0].astype(np.float32)
    alpha = (g[..., 3:] / 255.0) * region[a1:a2, b1:b2, None]
    fab[a1:a2, b1:b2] = fab[a1:a2, b1:b2] * (1 - alpha) + g[..., :3] * alpha  # the logo's exact colours
    return fab
