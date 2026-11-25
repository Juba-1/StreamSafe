"""The three cover strengths the streamer picks: light, balanced, high. Every place that shows or uses
them reads this module (settings, both windows, the analyzer), so they cannot drift apart.

A strength k multiplies the eye's operating threshold (thr_h6.txt; 0.23 for heads v3): thr = base * k.
Measured on the GTA 6 trailer 25 Nov (runs/trailer_scores_r3, thr 0.30 = k 1.3): bikini and underwear read
0.9 and up, modest close-ups 0.3-0.35 (covered by mistake), small far-away men up to 0.9 (covered by mistake).
"""

from __future__ import annotations

LIGHT, BALANCED, HIGH = 2.4, 1.95, 1.3  # thr ~0.55 / ~0.45 / ~0.30
LEVELS = (("s_light", LIGHT), ("s_mid", BALANCED), ("s_high", HIGH))  # (word key, k), as the buttons show them
DEFAULT = BALANCED
K_MIN, K_MAX = 1.0, 2.6  # what a saved or sent value is clamped to
# from this k up "score only" rules apply (risk.RiskPolicy.light): balanced and light. High keeps the
# careful rules (evidence build-up, early cover on the first looks, a lower bar for tiny people)
SCORE_ONLY_K = (HIGH + BALANCED) / 2
# the buttons before 3.0.0.50 (light 1.7, balanced 1.3 / 1.4, high 1.0): a value saved then moves to the same level now
OLD_LEVELS = ((1.7, LIGHT), (1.4, BALANCED), (1.3, BALANCED), (1.0, HIGH))


def clamp(k: float) -> float:
    return min(K_MAX, max(K_MIN, float(k)))


def level(k: float) -> str:
    """The word key of the level nearest to k."""
    return min(LEVELS, key=lambda lv: abs(lv[1] - k))[0]


def from_old(k: float) -> float:
    """A strength saved by an older version, as the same level today."""
    return min(OLD_LEVELS, key=lambda lv: abs(lv[0] - k))[1]
