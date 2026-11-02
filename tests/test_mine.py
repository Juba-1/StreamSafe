"""Hard-case mining (bench/mine_hard.py) on the stand-in models: band filter, de-duplication, outputs."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from streamsafe.config import Config
from tests.stubs import BLUE, make_video, stub_analyzer

_spec = importlib.util.spec_from_file_location("mine_hard", Path(__file__).parents[1] / "bench" / "mine_hard.py")
mine_hard = importlib.util.module_from_spec(_spec)
sys.modules["mine_hard"] = mine_hard  # dataclasses look their module up while the file runs
_spec.loader.exec_module(mine_hard)


@pytest.fixture
def video(tmp_path: Path) -> str:
    path = str(tmp_path / "borderline.mp4")
    make_video(
        path, n=40, turn_safe_at=0, turn_col=BLUE, texture=False
    )  # whole blobs, 230 px tall: a borderline (p 0.06) and a clearly safe person
    return path


def _miner(labeled: np.ndarray) -> mine_hard.Miner:
    return mine_hard.Miner(stub_analyzer(Config(), 1.0), labeled, fps=6, secs=10)


def test_keeps_one_borderline_person_and_writes_sheets(video: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mine_hard, "MID_H", 10_000)  # stub people are tall: treat them as "small / mid"
    miner = _miner(np.zeros((0, 16), np.float32))
    assert miner.mine(video, "train", cap=8) == 1  # same borderline person in every frame -> one crop
    assert miner.picked[0].cand.p == pytest.approx(0.06)  # the clearly safe person (p 0.02) is outside the band
    out = tmp_path / "out"
    miner.write(out)
    with open(out / "meta.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["split"] for r in rows] == ["train"]
    assert (out / "crops" / "00000.jpg").exists() and (out / "sheets" / "train_00.jpg").exists()
    assert np.load(out / "emb.npy").shape == (1, 16)


def test_skips_people_labeled_before(video: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mine_hard, "MID_H", 10_000)
    first = _miner(np.zeros((0, 16), np.float32))
    first.mine(video, "train", cap=8)
    again = _miner(first.picked[0].cand.emb[None].astype(np.float32))
    assert again.mine(video, "train", cap=8) == 0


def test_big_people_use_the_narrow_band(video: str) -> None:
    assert _miner(np.zeros((0, 16), np.float32)).mine(video, "train", cap=8) == 0  # 0.06 / 0.1746 = 0.34 < 0.45
