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
# RapidOCR builds its ONNX sessions with default options: one busy-spinning thread per CPU core. On the pod
# that starved the renderer (run 41: 98% late frames while OCR ran). The HUD reader gets OCR_THREADS
# threads that sleep when idle; it is a background signal, the picture must never wait for it.
OCR_THREADS = 2


class RapidHud:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.ocr = RapidOCR(**OCR_ARGS)
        holders = _session_holders(self.ocr)
        if not holders:
            raise RuntimeError("RapidOCR internals changed: no ONNX sessions found (pin rapidocr_onnxruntime==1.2.3)")
        for holder in holders:
            _limit_threads(holder)

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


def _session_holders(obj, depth: int = 0, seen: set | None = None) -> list:
    """Objects inside RapidOCR whose ``.session`` is an ONNX Runtime session (found by walking attributes,
    so a renamed attribute in another RapidOCR release does not break it)."""
    import onnxruntime as ort

    seen = set() if seen is None else seen
    if depth > 3 or id(obj) in seen or not hasattr(obj, "__dict__"):
        return []
    seen.add(id(obj))
    found = [obj] if isinstance(getattr(obj, "session", None), ort.InferenceSession) else []
    for value in vars(obj).values():
        found += _session_holders(value, depth + 1, seen)
    return found


def _limit_threads(holder) -> None:  # RapidOCR internals: its OrtInferSession wrapper (`.session` = ORT session)
    """Rebuild the holder's ONNX Runtime session with few, non-spinning threads (same model, same provider)."""
    import onnxruntime as ort

    old = holder.session
    opt = ort.SessionOptions()
    opt.log_severity_level = 4
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = OCR_THREADS
    opt.inter_op_num_threads = 1
    opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
    holder.session = ort.InferenceSession(old._model_path, sess_options=opt, providers=["CPUExecutionProvider"])
