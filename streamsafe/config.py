"""All engine settings, grouped by component.

Defaults are the best measured configuration (benchmark runs 20-27). Every field can be
overridden from the environment with its upper-case name, e.g. ``DELAY=1.5`` or
``EYE_EVERY=1``; field names are unique across sections so the mapping is unambiguous.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class DetectorConfig:
    det_model: str = "Nano"  # RF-DETR-Seg size: Nano / Small / Medium
    det_thr: float = 0.35  # detection confidence threshold
    min_h: int = 36  # people shorter than this (analysis px) are ignored: far away


@dataclass
class BackendConfig:
    # torch: PyTorch + CUDA (NVIDIA only; research and benchmarks). onnx: ONNX Runtime on NVIDIA (CUDA), any
    # DirectX 12 GPU on Windows (DirectML: NVIDIA, AMD, Intel) or the CPU - the product path
    backend: str = "torch"
    onnx_dir: str = ""  # folder written by tools/export_onnx.py (checked against its manifest.json at start)
    allow_cpu: bool = False  # tests only: without a GPU the onnx backend refuses to start instead of crawling


@dataclass
class EyeConfig:
    # Every downloaded model is pinned to an exact Hub commit (here, scene_revision, pose_revision): an upstream
    # change can never reach a streamer's PC unreviewed. The eye also runs the repo's own Python code
    # (trust_remote_code), so read that code before moving its pin.
    eye_repo: str = "nvidia/C-RADIOv4-H"  # NVIDIA Open Model License
    eye_revision: str = "0057b339059c0b9e1b4ba996f975410ebbfdfcc8"  # 2026-01-30
    eye_w: int = 224
    eye_h: int = 384
    head_dir: str = ""  # folder with fast_h6.txt, male_h6.txt, thr_h6.txt
    eye_every: int = 3  # a person followed by motion alone is re-checked every N decisions
    scene_prior: bool = True  # beach / pool scenes lower the threshold slightly
    full_res_crops: bool = True  # eye crops from the native frame (run36: women 63->77%, men 54->47%)


@dataclass
class RiskConfig:
    thr: float = 0.0  # operating threshold; 0 = read thr_h6.txt
    young_k: float = 1.6  # first look at a BIG person needs p >= young_k * thr
    young_min_h: int = 150  # ...for people at least this tall (analysis px)
    release_s: float = 1.0  # clear evidence needed this long before a person is uncovered
    # Golden rule (Ali): covering a modest person for a moment is better than uncovering a revealing one.
    # A person who looked clearly risky (max p >= fast_clear_k x thr) is never fast-cleared by two low readings
    # (a bent-over or half-off-screen pose reads low); 0 = old rule (only p >= 0.85 blocked it)
    fast_clear_k: float = 1.5
    # a new track where a covered person was lost moments ago (pose / edge broke the identity) takes over that
    # person's cover instead of starting from zero with the strict first-look rule (run 50: club dancer)
    handoff: bool = True
    handoff_iou: float = 0.4
    handoff_window: int = 5  # decisions (~0.5 s); never across a cut
    # a covered person the detector misses for a moment (bent over, half off screen, motion blur) keeps the
    # last garment, following the picture's motion, for this many decisions (~0.5 s); never across a cut.
    # Run 50: the club dancer at trailer 0:29.6 lost her cover when detection dropped her for 0.4 s
    coast_decisions: int = 5


@dataclass
class SceneConfig:
    scene_guard: bool = True  # whole-screen cover + game audio mute for explicit scenes
    scene_repo: str = "google/siglip2-base-patch16-224"  # Apache-2.0; also the beach / pool prior
    scene_revision: str = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"  # 2025-02-21
    scene_enter: float = 0.5  # scene model probability that turns the cover on (calibrated in run 41)
    scene_exit: float = 0.3  # ...and keeps it on
    scene_hold_s: float = 3.0  # cover stays this long after the last trigger
    scene_every_s: float = 0.5  # scene model period (and on every cut)
    scene_back_s: float = 0.5  # frames this far before a triggering decision are covered too (not shown yet)
    # covered people filling this share of the frame -> whole-screen card instead of garments. OFF (1.0) by
    # default: run 44 showed it replaces a good garment on a found person (trailer 0:56) - Ali prefers the
    # garment; strip-club / private-dance scenes are for the HUD reader and a trained scene head
    close_area: float = 1.0
    close_hold_s: float = 1.0
    hud_ocr: bool = False  # read the game's on-screen prompts (needs rapidocr_onnxruntime)
    hud_every_s: float = 0.5
    hud_hold_s: float = 4.0
    mute_lead_s: float = 0.3  # game audio is muted this early before a whole-screen cover ...
    mute_tail_s: float = 0.5  # ... and stays muted this long after it


@dataclass
class GarmentConfig:
    small_h: int = 64  # shorter people get a plain full cover (no pose); hysteresis x0.8 / x1.25
    pose_repo: str = "usyd-community/vitpose-base-simple"  # Apache-2.0
    pose_revision: str = "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"  # 2025-01-08
    kp_smooth: float = 0.5  # skeleton smoothing between consecutive decisions
    shape_memory: bool = True  # a garment never shrinks in a single step
    # fast movers (> margin_min_speed x their height between two decisions): the garment also covers where
    # they move to / come from, up to margin_max x height (Ali, run 47: volleyball runner partly outrun)
    motion_margin: bool = True
    margin_min_speed: float = 0.04
    margin_max: float = 0.3
    fabric_ds: int = 2  # cloth shading computed at 1/N resolution (outline at full resolution)
    color: str = "navy"  # one colour for everybody (streamer setting): a palette name or "#RRGGBB"
    logo: str = ""  # optional image on the chest (sponsor / ad mode), drawn in its own exact colours


@dataclass
class BrandConfig:
    """What viewers see on full-screen frames (fail-closed slate and scene card): the streamer's logo / ad."""

    slate_image: str = ""  # image centred on the slate (and the scene card unless scene_image is set)
    scene_image: str = ""
    slate_color: str = ""  # "#RRGGBB" background; empty = default (or the image's border colour)
    slate_text: str = "StreamSafe - back in a moment"  # empty = no text
    scene_text: str = "StreamSafe - scene hidden"


@dataclass
class RenderConfig:
    motion: bool = True  # every garment follows its own person to the frame being shown
    mc_min: float = 0.12  # minimum phase-correlation response to trust a motion estimate
    max_gap: int = 18  # a decision may protect frames at most this far away (same shot only)
    hold: int = 2  # a gap of at most this many frames repeats the last protected frame
    # steady placement (render/smooth.py): alpha-beta filter per person; numbers only, no image work
    smooth: bool = True  # run 47: shaking -40..47 % (p90), coverage unchanged, render +<1 ms
    smooth_alpha: float = 0.5  # share of the measurement taken each frame (1 = no smoothing)
    smooth_beta: float = 0.15  # velocity correction
    smooth_reset: float = 0.25  # a measurement farther than this (x garment size) from the prediction resets
    # > 0: a residual of smooth_fast x size or more is taken whole, smaller ones smoothed in proportion. 0 = off
    # (plain alpha). Run 49: 0.08 kept most of the shaking (trailer p90 0.187 vs 0.101 off) with no better cover;
    # fast movers are handled by the garment's motion margin instead
    smooth_fast: float = 0.0
    lookahead: bool = False  # a person turning risky is also covered in the earlier frames still waiting
    lookahead_frames: int = 60  # how far back (same shot, not yet shown)


@dataclass
class RuntimeConfig:
    mode: str = "realtime"  # realtime: analyzer in its own process, real clock; offline: deterministic
    step_every: int = 3  # offline mode analyses every N-th frame
    delay: float = 2.0  # stream delay in seconds: frame f is shown at capture_time(f) + delay
    fps: float = 30.0  # output frame rate
    secs: float = 1e9  # stop after this many seconds (file inputs)
    # a frame more than this past its due time is dropped instead of shown, so a slow output (encoder,
    # busy CPU) never makes the stream delay - and the frames waiting in memory - grow without limit
    max_late_s: float = 0.5
    live_log_s: float = 0.0  # > 0: print a "LIVE {...}" progress line this often (soak tests, support logs)


@dataclass
class OutputConfig:
    stats_out: str = ""  # optional JSON with timings / counters
    diag_out: str = ""  # optional JSON log of every decision and every drawn garment
    obs_state: str = ""  # state file for the OBS guard script (obs/streamsafe_guard.lua): heartbeat + mute


@dataclass
class Config:
    backend: BackendConfig = field(default_factory=BackendConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    brand: BrandConfig = field(default_factory=BrandConfig)
    garment: GarmentConfig = field(default_factory=GarmentConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @staticmethod
    def from_env(env: dict[str, str] | None = None, **overrides: Any) -> Config:
        """Defaults <- environment variables <- keyword overrides (by flat field name)."""
        env = os.environ if env is None else env
        cfg = Config()
        for section, name in _flat_fields(cfg):
            raw = env.get(name.upper())
            if raw is not None:
                setattr(section, name, _parse(raw, type(getattr(section, name))))
        for name, value in overrides.items():
            cfg.set(name, value)
        return cfg

    def set(self, name: str, value: Any) -> None:
        for section, field_name in _flat_fields(self):
            if field_name == name:
                setattr(section, name, value)
                return
        raise KeyError(f"unknown setting: {name}")


def _flat_fields(cfg: Config) -> list[tuple[Any, str]]:
    out = []
    for f in fields(cfg):
        section = getattr(cfg, f.name)
        assert is_dataclass(section)
        out.extend((section, sf.name) for sf in fields(section))
    names = [n for _, n in out]
    assert len(names) == len(set(names)), "setting names must be unique across sections"
    return out


def _parse(raw: str, kind: type) -> Any:
    if kind is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return kind(raw)
