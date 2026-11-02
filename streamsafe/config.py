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
class EyeConfig:
    eye_repo: str = "nvidia/C-RADIOv4-H"
    eye_w: int = 224
    eye_h: int = 384
    head_dir: str = ""  # folder with fast_h6.txt, male_h6.txt, thr_h6.txt
    eye_every: int = 3  # a person followed by motion alone is re-checked every N decisions
    scene_prior: bool = True  # beach / pool scenes lower the threshold slightly


@dataclass
class RiskConfig:
    thr: float = 0.0  # operating threshold; 0 = read thr_h6.txt
    young_k: float = 1.6  # first look at a BIG person needs p >= young_k * thr
    young_min_h: int = 150  # ...for people at least this tall (analysis px)
    release_s: float = 1.0  # clear evidence needed this long before a person is uncovered


@dataclass
class GarmentConfig:
    small_h: int = 64  # shorter people get a plain full cover (no pose); hysteresis x0.8 / x1.25
    kp_smooth: float = 0.5  # skeleton smoothing between consecutive decisions
    shape_memory: bool = True  # a garment never shrinks in a single step
    fabric_ds: int = 2  # cloth shading computed at 1/N resolution (outline at full resolution)
    color: str = "navy"  # one colour for everybody (streamer setting)
    logo: str = ""  # optional PNG on the chest (sponsor / ad mode)


@dataclass
class RenderConfig:
    motion: bool = True  # every garment follows its own person to the frame being shown
    mc_min: float = 0.12  # minimum phase-correlation response to trust a motion estimate
    max_gap: int = 18  # a decision may protect frames at most this far away (same shot only)
    hold: int = 2  # a gap of at most this many frames repeats the last protected frame


@dataclass
class RuntimeConfig:
    mode: str = "realtime"  # realtime: analyzer in its own process, real clock; offline: deterministic
    step_every: int = 3  # offline mode analyses every N-th frame
    delay: float = 2.0  # stream delay in seconds: frame f is shown at capture_time(f) + delay
    fps: float = 30.0  # output frame rate
    secs: float = 1e9  # stop after this many seconds (file inputs)


@dataclass
class OutputConfig:
    stats_out: str = ""  # optional JSON with timings / counters
    diag_out: str = ""  # optional JSON log of every decision and every drawn garment


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
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
