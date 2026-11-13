"""Text reader: PaddleOCR's PP-OCR detector + English recogniser as ONNX files (Apache-2.0, converted by
RapidOCR), run with ONNX Runtime on the CPU with few, non-spinning threads. The pre- and post-processing here is
plain numpy / OpenCV, so no OCR package (and no pyclipper / shapely) has to ship.

``TextReader.read(bgr)`` -> ``[Line(box, text, conf)]`` for the text found in one image (a region cut from the
frame). Boxes are axis-aligned (x0, y0, x1, y1) in that image's pixels; game text is horizontal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DET_FILE = "det.onnx"  # ch_PP-OCRv4_det_infer.onnx
REC_FILE = "rec.onnx"  # en_PP-OCRv3_rec_infer.onnx (its character list is inside the file)

DET_LIMIT = 960  # long side of the detector input (a multiple of 32)
DET_THRESH = 0.3  # probability map -> binary
BOX_THRESH = 0.6  # mean probability inside a box
UNCLIP = 1.6  # DB post-processing: boxes are grown by area * UNCLIP / perimeter
MIN_BOX = 5  # smaller boxes (px) are noise
REC_H = 48
REC_MAX_W = 1600
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class Line:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    text: str
    conf: float


def _session(path: Path, threads: int):
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.log_severity_level = 4
    opt.intra_op_num_threads = threads
    opt.inter_op_num_threads = 1
    opt.add_session_config_entry("session.intra_op.allow_spinning", "0")  # idle threads sleep: the picture never waits for us
    return ort.InferenceSession(str(path), sess_options=opt, providers=["CPUExecutionProvider"])


class TextReader:
    def __init__(self, folder: Path, threads: int = 2) -> None:
        self.det = _session(folder / DET_FILE, threads)
        self.rec = _session(folder / REC_FILE, threads)
        chars = self.rec.get_modelmeta().custom_metadata_map["character"].split("\n")
        if chars and chars[-1] == "":
            chars = chars[:-1]
        n_out = self.rec.get_outputs()[0].shape[-1]
        # CTC classes: 0 = blank, then the character list, then (when the model has one more class) a space
        self.chars = ["", *chars]
        if isinstance(n_out, int) and n_out > len(self.chars):
            self.chars += [" "] * (n_out - len(self.chars))
        self._det_in = self.det.get_inputs()[0].name
        self._rec_in = self.rec.get_inputs()[0].name

    # ------------------------------------------------------------------------------------------ detection
    def boxes(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        scale = min(1.0, DET_LIMIT / max(h, w))
        nh, nw = max(32, round(h * scale / 32) * 32), max(32, round(w * scale / 32) * 32)
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        x = (img[:, :, ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD
        prob = self.det.run(None, {self._det_in: x.transpose(2, 0, 1)[None]})[0][0, 0]
        mask = (prob > DET_THRESH).astype(np.uint8)
        found = []
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours[:200]:
            rect = cv2.minAreaRect(c)
            _centre, (rw, rh), _angle = rect
            if min(rw, rh) < 3:
                continue
            x0, y0, bw, bh = cv2.boundingRect(c)
            if float(prob[y0 : y0 + bh, x0 : x0 + bw][mask[y0 : y0 + bh, x0 : x0 + bw] > 0].mean()) < BOX_THRESH:
                continue
            grow = rw * rh * UNCLIP / (2 * (rw + rh))  # DB "unclip", for the box's own rectangle
            ex0, ey0, ex1, ey1 = x0 - grow, y0 - grow, x0 + bw + grow, y0 + bh + grow
            box = (
                max(0, math.floor(ex0 / nw * w)),
                max(0, math.floor(ey0 / nh * h)),
                min(w, math.ceil(ex1 / nw * w)),
                min(h, math.ceil(ey1 / nh * h)),
            )
            if box[2] - box[0] >= MIN_BOX and box[3] - box[1] >= MIN_BOX:
                found.append(box)
        return found

    # ---------------------------------------------------------------------------------------- recognition
    def recognise(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(1, crops[i].shape[0]))
        for i in order:  # one crop per call: widths differ, and a subtitle band has few lines
            crop = crops[i]
            h, w = crop.shape[:2]
            width = min(REC_MAX_W, max(16, math.ceil(REC_H * w / h)))
            img = cv2.resize(crop, (width, REC_H), interpolation=cv2.INTER_LINEAR)
            x = (img[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
            probs = self.rec.run(None, {self._rec_in: x.transpose(2, 0, 1)[None]})[0][0]
            idx = probs.argmax(axis=1)
            best = probs.max(axis=1)
            keep = (idx != 0) & np.concatenate([[True], idx[1:] != idx[:-1]])
            text = "".join(self.chars[k] if k < len(self.chars) else "" for k in idx[keep])
            out[i] = (text, float(best[keep].mean()) if keep.any() else 0.0)
        return out

    def read(self, bgr: np.ndarray, pad: int = 4) -> list[Line]:
        """Text in ``bgr`` (cut from the frame; text touching the image edge is missed, so cut with some room)."""
        h, w = bgr.shape[:2]
        boxes = self.boxes(bgr)
        crops = []
        for x0, y0, x1, y1 in boxes:
            crops.append(bgr[max(0, y0 - pad // 2) : min(h, y1 + pad // 2), max(0, x0 - pad) : min(w, x1 + pad)])
        lines = []
        for box, (text, conf) in zip(boxes, self.recognise(crops), strict=True):
            if text.strip():
                lines.append(Line(box, text, conf))
        return sorted(lines, key=lambda ln: (ln.box[1] // 8, ln.box[0]))
