"""HUD reader: RapidOCR (ONNX Runtime, Apache-2.0) on the screen areas where games show their prompts.

Only two small regions are read (top-left help box, bottom subtitles), each padded with a black border
(RapidOCR misses text that touches the image edge). The regions are GTA V's; GTA 6 positions get set from
the first real footage (one constant here, nothing else changes).
"""

from __future__ import annotations

import cv2
import numpy as np

# (y0, y1, x0, x1) as fractions of the frame
REGIONS = ((0.0, 0.22, 0.0, 0.42), (0.84, 0.97, 0.2, 0.8))
READ_W = 1280  # regions are cut from the frame resized to this width (text stays readable, cost bounded)
PAD = (60, 60, 40, 40)
# RapidOCR's default text detector upscales the SHORT side to 736 px (a 260 px tall region becomes 2264 x 736)
# and runs an angle classifier: 0.45 s per region on 2 CPU cores. Capping the LONG side at 640 and dropping
# the angle step (HUD text is never upside down) reads the same words in 0.1 s (TESTED, 7 Nov 2026).
OCR_ARGS = dict(det_limit_side_len=640, det_limit_type="max", det_model_path=None, use_angle_cls=False)


class RapidHud:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.ocr = RapidOCR(**OCR_ARGS)

    def __call__(self, bgr: np.ndarray) -> str:
        h, w = bgr.shape[:2]
        if w != READ_W:
            bgr = cv2.resize(bgr, (READ_W, round(h * READ_W / w)), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        words = []
        for y0, y1, x0, x1 in REGIONS:
            roi = bgr[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]
            res, _ = self.ocr(cv2.copyMakeBorder(roi, *PAD, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
            words += [r[1] for r in res or []]
        return " ".join(words)
