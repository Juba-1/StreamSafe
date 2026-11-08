"""Linear risk heads on top of the eye embedding (numpy only).

File format (one line): ``scale;bias;w1,w2,...`` with integer-quantised weights ``w / scale``.
p(cover) = sigmoid(E.w_fast + b_fast) * (1 - sigmoid(E.w_male + b_male))  -- gated: a man is never "risky".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FAST_FILE, MALE_FILE, THR_FILE = "fast_h6.txt", "male_h6.txt", "thr_h6.txt"
SCENE_FILE = "scene_h.txt"  # optional: explicit-scene head on the scene model's image embedding


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


@dataclass(frozen=True)
class LinearHead:
    w: np.ndarray
    b: float

    @staticmethod
    def load(path: Path) -> LinearHead:
        scale, bias, weights = path.read_text().split(";")
        return LinearHead(np.array(weights.split(","), np.float32) / float(scale), float(bias))

    def __call__(self, emb: np.ndarray) -> np.ndarray:
        return _sigmoid(emb @ self.w + self.b)


@dataclass(frozen=True)
class RiskHeads:
    fast: LinearHead
    male: LinearHead
    thr: float

    @staticmethod
    def load(head_dir: str | Path) -> RiskHeads:
        d = Path(head_dir)
        heads = RiskHeads(LinearHead.load(d / FAST_FILE), LinearHead.load(d / MALE_FILE), float((d / THR_FILE).read_text()))
        heads.check()
        return heads

    def check(self) -> None:
        """Refuse damaged or mismatched head files at start-up, instead of failing on every frame later."""
        ok = (
            len(self.fast.w) == len(self.male.w) > 0
            and np.isfinite(self.fast.w).all()
            and np.isfinite(self.male.w).all()
            and np.isfinite([self.fast.b, self.male.b]).all()
            and 0.0 < self.thr < 1.0
        )
        if not ok:
            raise ValueError(f"invalid risk heads (sizes {len(self.fast.w)}/{len(self.male.w)}, thr {self.thr})")

    def prisk(self, emb: np.ndarray) -> np.ndarray:
        return self.fast(emb) * (1 - self.male(emb))
