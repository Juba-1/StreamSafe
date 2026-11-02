"""Hard-case mining for a labeling round: the people the eye is least sure about.

For every video the engine's own detector + eye crops (``Analyzer.detect``) are scored; a person is a
candidate when its risk p is near the threshold - small and mid-size people (the known recall gap)
get a wider band. Candidates are de-duplicated (inside the video and against every crop that is
already labeled), the hardest are kept, and numbered contact sheets are drawn for labeling.

usage: python bench/mine_hard.py OUT_DIR HEAD_DIR LABELED_EMB.npz VIDEO [VIDEO ...]
env:   FPS=2  SECS=90  PER_VIDEO=8  PER_TEST_VIDEO=40  TEST=name,name  (held-out videos -> test split)

Output: OUT_DIR/{crops,ctx}/<id>.jpg, meta.csv (id,split,video,frame,h,p), emb.npy (row = id),
        sheets/<split>_<k>.jpg (24 numbered cells: isolated crop | the person in context).
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.config import Config
from streamsafe.io.sources import VideoFileSource
from streamsafe.models import build_models
from streamsafe.types import ANALYSIS_SIZE

MID_H = 200  # analysis px: below this a person is "small / mid" (wider band, preferred)
BAND_SMALL = (0.2, 3.0)  # p / thr band for small and mid people
BAND_BIG = (0.45, 2.2)  # p / thr band for big people (borderline outfits, modest uniforms)
SMALL_SHARE = 0.7
DUP_VIDEO = 0.93  # cosine: same person / pose inside one video
DUP_LABELED = 0.95  # cosine: already labeled in an earlier round
CELL_H, CROP_W, CTX_W, ID_H = 180, 120, 220, 28
COLS, ROWS = 4, 6


@dataclass
class Candidate:
    hardness: float  # |log(p / thr)|: 0 = exactly on the threshold
    emb: np.ndarray
    crop: np.ndarray  # RGB, isolated as the eye sees it
    ctx: np.ndarray  # BGR, the person in the frame with a box
    frame: int
    h: int
    p: float


@dataclass
class Picked:
    split: str
    video: str
    cand: Candidate


@dataclass
class Miner:
    analyzer: Analyzer
    labeled: np.ndarray  # (n, d) float32, L2-normalised
    fps: float
    secs: float
    picked: list[Picked] = field(default_factory=list)

    def mine(self, path: str, split: str, cap: int) -> int:
        name = Path(path).stem
        try:
            source = VideoFileSource(path, self.fps, self.secs, size=ANALYSIS_SIZE)
        except FileNotFoundError as exc:
            print("SKIP", name, exc, flush=True)
            return 0
        eye, thr = self.analyzer.models.eye, self.analyzer.thr
        cands: list[Candidate] = []
        for k, frame in enumerate(source.frames()):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            people, crops = self.analyzer.detect(rgb)
            if not people:
                continue
            emb = eye.embed(crops)
            for person, crop, e, p in zip(people, crops, emb, eye.prisk(emb), strict=True):
                lo, hi = BAND_SMALL if person.height < MID_H else BAND_BIG
                if lo * thr <= p <= hi * thr:
                    cands.append(Candidate(abs(float(np.log(p / thr))), e, crop, _context(frame, person.box), k, person.height, float(p)))
        source.close()
        kept = self._select(cands, cap)
        self.picked += [Picked(split, name, c) for c in kept]
        print("MINED", split, name, "candidates", len(cands), "kept", len(kept), flush=True)
        return len(kept)

    def _select(self, cands: list[Candidate], cap: int) -> list[Candidate]:
        """Hardest first, small/mid people preferred, no near-duplicates."""
        cands.sort(key=lambda c: c.hardness)
        if cands and len(self.labeled):
            sims = np.stack([c.emb for c in cands]).astype(np.float32) @ self.labeled.T
            cands = [c for c, s in zip(cands, sims.max(1), strict=True) if s < DUP_LABELED]
        kept: list[Candidate] = []

        def take(pool: list[Candidate], limit: int) -> None:
            for c in pool:
                if len(kept) >= limit:
                    return
                if all(float(c.emb @ k.emb) < DUP_VIDEO for k in kept):  # also rejects ones already kept
                    kept.append(c)

        small = [c for c in cands if c.h < MID_H]
        take(small, round(cap * SMALL_SHARE))
        take([c for c in cands if c.h >= MID_H], cap)
        take(small, cap)  # big people ran out: fill with more small / mid ones
        return kept

    def write(self, out: Path) -> None:
        for sub in ("crops", "ctx", "sheets"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        with open(out / "meta.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "split", "video", "frame", "h", "p"])
            for i, pk in enumerate(self.picked):
                c = pk.cand
                cv2.imwrite(str(out / "crops" / f"{i:05d}.jpg"), cv2.cvtColor(c.crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(str(out / "ctx" / f"{i:05d}.jpg"), c.ctx, [cv2.IMWRITE_JPEG_QUALITY, 85])
                w.writerow([i, pk.split, pk.video, c.frame, c.h, round(c.p, 4)])
        np.save(out / "emb.npy", np.stack([pk.cand.emb for pk in self.picked]).astype(np.float16))
        for split in ("train", "test"):
            ids = [i for i, pk in enumerate(self.picked) if pk.split == split]
            for s in range(0, len(ids), COLS * ROWS):
                sheet = _sheet([(i, self.picked[i].cand) for i in ids[s : s + COLS * ROWS]])
                cv2.imwrite(str(out / "sheets" / f"{split}_{s // (COLS * ROWS):02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("WROTE", len(self.picked), "crops", flush=True)


def _context(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, max(w, h) * 1.4
    X1, Y1 = max(0, int(cx - r)), max(0, int(cy - r))
    X2, Y2 = min(frame.shape[1], int(cx + r)), min(frame.shape[0], int(cy + r))
    ctx = frame[Y1:Y2, X1:X2].copy()
    cv2.rectangle(ctx, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (0, 255, 255), 1)
    return ctx


def _fit(im: np.ndarray, width: int) -> np.ndarray:
    """Resize to CELL_H x <= width, keeping the aspect ratio, on a dark panel."""
    s = min(CELL_H / im.shape[0], width / im.shape[1])
    small = cv2.resize(im, (max(1, round(im.shape[1] * s)), max(1, round(im.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    panel = np.full((CELL_H, width, 3), 40, np.uint8)
    y, x = (CELL_H - small.shape[0]) // 2, (width - small.shape[1]) // 2
    panel[y : y + small.shape[0], x : x + small.shape[1]] = small
    return panel


def _sheet(cells: list[tuple[int, Candidate]]) -> np.ndarray:
    cw, chh = CROP_W + CTX_W + 6, CELL_H + ID_H
    sheet = np.full((ROWS * chh, COLS * cw, 3), 20, np.uint8)
    for n, (i, c) in enumerate(cells):
        y, x = (n // COLS) * chh, (n % COLS) * cw
        sheet[y + ID_H : y + chh, x : x + CROP_W] = _fit(cv2.cvtColor(c.crop, cv2.COLOR_RGB2BGR), CROP_W)
        sheet[y + ID_H : y + chh, x + CROP_W + 2 : x + CROP_W + 2 + CTX_W] = _fit(c.ctx, CTX_W)
        cv2.putText(sheet, f"{i}  h{c.h}", (x + 4, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return sheet


def main(argv: list[str]) -> None:
    out, head_dir, labeled_npz, videos = Path(argv[0]), argv[1], argv[2], argv[3:]
    labeled = np.load(labeled_npz)["X"].astype(np.float32)
    cfg = Config.from_env(head_dir=head_dir, scene_prior=False)
    analyzer = Analyzer(cfg, build_models(cfg), output_scale=1.0)
    miner = Miner(analyzer, labeled, float(os.environ.get("FPS", 2)), float(os.environ.get("SECS", 90)))
    test = set(filter(None, os.environ.get("TEST", "").split(",")))
    for path in videos:
        split = "test" if Path(path).stem in test else "train"
        miner.mine(path, split, int(os.environ.get("PER_TEST_VIDEO" if split == "test" else "PER_VIDEO", 40 if split == "test" else 8)))
    miner.write(out)


if __name__ == "__main__":
    main(sys.argv[1:])
