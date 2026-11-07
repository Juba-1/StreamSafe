"""Scene layer: whole-screen cover for scenes no garment can handle.

A person garment needs a detected person. Strip-club dances, sex scenes and extreme close-ups of a body
often show no detectable "person" at all, so the whole frame is covered instead (and the game audio is
muted with it, see ``runtime/gate.py``). Three independent signals, any one is enough:

  * SCENE: a trained explicit-scene head on the scene model's image embedding (``scene_h.txt``), checked
    every ``scene_every_s`` and on every cut; without a trained head this signal is off (zero-shot prompts
    were measured unreliable, see ``models/torch_backend/scene.py``);
  * CLOSE: the people the eye wants covered fill at least ``close_area`` of the frame (body close-ups:
    a garment over most of the screen is both ugly and leak-prone at its edges), hold ``close_hold_s``;
  * HUD:   on-screen text of the game's own prompts ("end the dance", "private room" ...), read by an
    optional OCR model; the keyword match below is plain code.

The latch (deterministic): ON at p >= ``scene_enter`` or a HUD hit; it stays ON while p >= ``scene_exit``,
or while the last trigger is younger than its hold time. Hysteresis + hold = no flicker on a noisy score.
"""

from __future__ import annotations

import re

from streamsafe.config import SceneConfig

# Game prompts that only appear during a private dance / strip-club interaction (letters only, lower case).
HUD_KEYWORDS = (
    "endthedance",
    "stoptouching",
    "movethecamera",
    "privateroom",
    "lapdance",
    "privatedance",
    "approachadancer",
    "acceptadance",
    "likemeter",
    "touchher",
    "totouch",
)


def hud_hit(text: str) -> str:
    """The first HUD keyword in ``text`` ('' when none). Spaces / punctuation / case are ignored, so OCR
    that splits or joins words still matches."""
    norm = re.sub("[^a-z]", "", text.lower())
    return next((k for k in HUD_KEYWORDS if k in norm), "")


class SceneLatch:
    """Covered / not-covered state of the whole scene. ``update`` once per decision; readings are optional
    (a decision that did not run the scene model or the OCR passes None)."""

    def __init__(self, cfg: SceneConfig) -> None:
        self.cfg = cfg
        self.on = False
        self._last_scene = -1e9  # time of the last scene-model trigger
        self._last_hud = -1e9
        self._last_close = -1e9

    def mark_close(self, now: float) -> None:
        self._last_close = now
        self.on = True

    def update(self, now: float, p_explicit: float | None = None, hud_at: float | None = None) -> bool:
        """``p_explicit``: a scene-model reading of this decision's frame; ``hud_at``: stream time of a frame
        where the HUD reader found a keyword (reads finish later, so it can be in the past)."""
        cfg = self.cfg
        if p_explicit is not None and (p_explicit >= cfg.scene_enter or (self.on and p_explicit >= cfg.scene_exit)):
            self._last_scene = now
        if hud_at is not None:
            self._last_hud = max(self._last_hud, hud_at)
        scene_on = now - self._last_scene <= cfg.scene_hold_s
        hud_on = now - self._last_hud <= cfg.hud_hold_s
        close_on = now - self._last_close <= cfg.close_hold_s
        self.on = scene_on or hud_on or close_on
        return self.on
