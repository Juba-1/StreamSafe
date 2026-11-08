"""Full-screen frames: the fail-closed slate ("back in a moment") and the scene card (explicit scene hidden).

Both can carry the streamer's own image (logo / ad) on a background colour - ``BrandConfig``.
"""

from __future__ import annotations

import cv2
import numpy as np

from streamsafe.config import BrandConfig
from streamsafe.garment.fabric import load_logo, parse_color

BACKGROUND = (40, 28, 22)  # BGR
SLATE_TEXT = "StreamSafe - back in a moment"
SCENE_BACKGROUND = (52, 36, 30)
SCENE_TEXT = "StreamSafe - scene hidden"
IMAGE_HEIGHT = 0.62  # share of the frame height the image may use (with text: a little less)


def border_color(image: np.ndarray) -> tuple[int, int, int]:
    """Median BGR of the image's outer ring of pixels (opaque ones): the natural background around it."""
    ring = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    if image.shape[2] == 4 and (ring[:, 3] > 128).any():
        ring = ring[ring[:, 3] > 128]
    return tuple(int(v) for v in np.median(ring[:, :3], axis=0))


def make_slate(
    width: int,
    height: int,
    text: str = SLATE_TEXT,
    background: tuple[int, int, int] | None = BACKGROUND,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Opaque frame: ``background`` (or the image's border colour), the image centred (exact colours), text below."""
    if background is None:
        background = border_color(image) if image is not None else BACKGROUND
    slate = np.full((height, width, 3), background, np.uint8)
    scale = 1.6 * width / 1920
    thick = max(1, round(3 * width / 1920))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick) if text else ((0, 0), 0)
    text_y = (height + th) // 2
    if image is not None:
        box_h = int(height * (IMAGE_HEIGHT - (0.1 if text else 0)))
        s = min(box_h / image.shape[0], 0.9 * width / image.shape[1])
        iw, ih = max(1, int(image.shape[1] * s)), max(1, int(image.shape[0] * s))
        img = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        y0 = (height - ih - (int(2.2 * th) if text else 0)) // 2
        x0 = (width - iw) // 2
        a = img[..., 3:].astype(np.float32) / 255.0
        roi = slate[y0 : y0 + ih, x0 : x0 + iw].astype(np.float32)
        slate[y0 : y0 + ih, x0 : x0 + iw] = (roi * (1 - a) + img[..., :3] * a).astype(np.uint8)
        text_y = y0 + ih + int(1.6 * th)
    if text:
        lum = 0.114 * background[0] + 0.587 * background[1] + 0.299 * background[2]
        ink = (235, 235, 235) if lum < 140 else (30, 30, 30)
        cv2.putText(slate, text, ((width - tw) // 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, thick)
    return slate


def make_scene_card(width: int, height: int) -> np.ndarray:
    return make_slate(width, height, SCENE_TEXT, SCENE_BACKGROUND)


def brand_frames(width: int, height: int, brand: BrandConfig) -> tuple[np.ndarray, np.ndarray]:
    """(slate, scene card) for this streamer's brand settings. No image and no colour -> StreamSafe defaults."""
    slate_img = load_logo(brand.slate_image)
    scene_img = load_logo(brand.scene_image) if brand.scene_image else slate_img
    color = parse_color(brand.slate_color) if brand.slate_color else None
    slate_bg = color or (None if slate_img is not None else BACKGROUND)
    scene_bg = color or (None if scene_img is not None else SCENE_BACKGROUND)
    return (
        make_slate(width, height, brand.slate_text, slate_bg, slate_img),
        make_slate(width, height, brand.scene_text, scene_bg, scene_img),
    )
