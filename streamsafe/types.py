"""Data passed between the engine's stages.

Coordinates: the analyzer works on a fixed-size *analysis frame* (``ANALYSIS_SIZE``); output
layers are already scaled to the output frame. Boxes are ``(x1, y1, x2, y2)`` integer pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ANALYSIS_SIZE = (960, 540)  # (width, height) every frame is analysed at
Box = tuple[int, int, int, int]


@dataclass(slots=True)
class Person:
    """One detected person in one analysed frame."""

    box: Box
    mask: np.ndarray  # bool, analysis-frame size
    emb: np.ndarray | None = None  # L2-normalised eye embedding
    p: float = 0.0  # eye risk probability for this frame
    tid: int = -1  # track id, set by the tracker
    risky: bool = False  # cover decision, set by the risk policy
    stale: bool = False  # eye skipped: emb / p carried over from the matched track
    hint: int | None = None  # track id proposed by motion matching (eye skipped)

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass(slots=True)
class MotionRef:
    """The person's own pixels at decision time, used to move the garment to later frames."""

    region: Box  # analysis-frame window the reference was cut from
    ref: np.ndarray  # float32 grayscale, background flattened, at most 128 px
    size: tuple[int, int]  # (w, h) of ref
    window: np.ndarray  # Hanning window for phase correlation


@dataclass(slots=True)
class Layer:
    """A ready-to-draw garment for one covered person, in output-frame pixels."""

    tid: int
    y: int  # top-left corner in the output frame
    x: int
    mask: np.ndarray  # bool (h, w)
    fabric: np.ndarray  # uint8 (h, w, 3) BGR
    centroid: tuple[float, float]  # (y, x) in output pixels
    motion: MotionRef | None = None
    _mask_u8: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def mask_u8(self) -> np.ndarray:
        """Zero-copy uint8 view of the mask (what cv2.copyTo needs)."""
        if self._mask_u8 is None:
            self._mask_u8 = self.mask.view(np.uint8)
        return self._mask_u8

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape  # type: ignore[return-value]


@dataclass(slots=True)
class Decision:
    """The analyzer's verdict for one frame. ``ok=False`` means the frame failed: fail closed."""

    frame_id: int
    ok: bool
    layers: list[Layer] = field(default_factory=list)
    n_people: int = 0
    n_risky: int = 0
    people: list[tuple[int, Box]] = field(default_factory=list)  # (track id, box) of everyone seen
    scene: bool = False  # whole-screen cover (explicit scene, see analysis/scene.py); layers are then empty
    scene_from: int | None = None  # earliest frame the scene evidence came from (a HUD read finishes later)
    backfilled: int = 0  # garments added later by look-ahead
    refs: dict[int, MotionRef] = field(default_factory=dict)  # look-ahead only: motion reference per track id
