"""The refactored engine must make exactly the decisions and draw exactly the pixels of the previous
engine (engine2) in deterministic mode. Skipped when the reference copy is not available."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from streamsafe.config import Config
from streamsafe.io.sources import VideoFileSource
from streamsafe.runtime.runner import run
from tests.stubs import BLUE, GREEN, make_video, stub_analyzer

REF = Path(os.environ.get("STREAMSAFE_REF", Path(__file__).resolve().parents[2] / "engine2_ref"))
pytestmark = pytest.mark.skipif(not (REF / "streamsafe").is_dir(), reason="reference engine not available")

VIDEOS = {
    "walk_with_cut": {},
    "turns_safe": {"n": 150, "cut_at": 10**6, "turn_safe_at": 60, "turn_col": GREEN},
    "turns_borderline": {"n": 150, "cut_at": 10**6, "turn_safe_at": 60, "turn_col": BLUE},
    "flat": {"texture": False},
    "flash": {"flash_at": 40},
}


class HashSink:
    def __init__(self) -> None:
        self.hashes: list[str] = []

    def write(self, frame) -> None:
        self.hashes.append(hashlib.md5(frame.tobytes()).hexdigest())

    def close(self) -> None:
        pass


def reference(video: str) -> dict:
    out = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ref_harness.py")), str(REF), video],
        capture_output=True, text=True, check=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    ).stdout  # fmt: skip
    line = next(line for line in out.splitlines() if line.startswith("REF_RESULT "))
    return json.loads(line[len("REF_RESULT ") :])


@pytest.mark.parametrize("name", list(VIDEOS))
def test_same_decisions_and_pixels_as_engine2(tmp_path, name):
    video = str(tmp_path / f"{name}.mp4")
    make_video(video, **VIDEOS[name])
    ref = reference(video)

    diag = str(tmp_path / "diag.json")
    cfg = Config.from_env(env={}, mode="offline", diag_out=diag, smooth=False)  # engine2 had no steadying
    source, sink = VideoFileSource(video, 30.0, size=(1920, 1080)), HashSink()
    run(cfg, source, sink, factory=stub_analyzer)
    with open(diag) as f:
        new = json.load(f)

    assert [d[:2] for d in new["dec"]] == ref["diag"]["dec"], "decisions differ"  # (+ prior, threshold now)
    assert new["frm"] == ref["diag"]["frm"], "drawn garments differ"
    assert sink.hashes == ref["hashes"], "output pixels differ"
