"""RF-DETR-Seg person detector (Apache-2.0 sizes N-L), fp16 on the GPU."""

from __future__ import annotations

import numpy as np
from PIL import Image

from streamsafe.models.torch_backend.device import DEVICE, HALF
from streamsafe.types import Box

COCO_PERSON = 1


class RFDETRDetector:
    def __init__(self, size: str = "Nano", thr: float = 0.35) -> None:
        import rfdetr

        self.model = getattr(rfdetr, "RFDETRSeg" + size)()
        self.thr = thr
        self.precision = "fp32"
        if DEVICE == "cuda":
            try:
                self.model.optimize_for_inference(compile=False, dtype=HALF)
                self.precision = "fp16"
            except Exception as exc:  # older rfdetr builds: keep fp32, record why
                self.precision = f"fp32 ({exc!r:.60})"

    def __call__(self, rgb: np.ndarray) -> list[tuple[Box, np.ndarray]]:
        det = self.model.predict(Image.fromarray(rgb), threshold=self.thr)
        if det.mask is None or len(det.xyxy) == 0:
            return []
        cls = np.asarray(det.class_id) if det.class_id is not None else np.zeros(len(det.xyxy))
        # class ids are 1-based COCO in the published weights; single-class fine-tunes use 0
        keep = np.where(cls == COCO_PERSON)[0] if (cls == COCO_PERSON).any() else np.where(cls == 0)[0]
        return [(tuple(int(v) for v in det.xyxy[i]), det.mask[i].astype(bool)) for i in keep]
