"""ONNX backend safety checks that need no model files: the manifest gate and the no-silent-CPU rule."""

import json

import numpy as np
import pytest

from streamsafe.models.onnx_backend import REQUIRED, sha256, verify
from streamsafe.models.onnx_backend.detector import _postprocess

PINS = {"org/eye": "abc"}


def _folder(tmp_path, revision="abc"):
    files = {}
    for name in REQUIRED:
        (tmp_path / name).write_bytes(name.encode())
        files[name] = {"sha256": sha256(tmp_path / name), "bytes": len(name)}
    files["eye.onnx"].update(repo="org/eye", revision=revision)
    (tmp_path / "manifest.json").write_text(json.dumps({"files": files}))
    return tmp_path


def test_good_folder_passes(tmp_path):
    assert set(verify(_folder(tmp_path), PINS)) == set(REQUIRED)


def test_damaged_file_stops_the_engine(tmp_path):
    folder = _folder(tmp_path)
    (folder / "pose.onnx").write_bytes(b"pose.onnY")  # same size, other bytes
    with pytest.raises(ValueError, match="damaged"):
        verify(folder, PINS)


def test_wrong_pin_stops_the_engine(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        verify(_folder(tmp_path, revision="old"), PINS)


def test_missing_manifest(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        verify(tmp_path, PINS)


def test_no_silent_cpu(tmp_path, monkeypatch):
    from streamsafe.models.onnx_backend import session

    monkeypatch.setattr(session, "providers", lambda: [session.PROVIDERS["cpu"]])
    with pytest.raises(RuntimeError, match="no usable GPU"):
        session.open_session(tmp_path / "x.onnx")


def test_postprocess_keeps_people_scaled_and_clamped():
    logits = np.full((3, 91), -10.0, np.float32)
    logits[0, 1] = 5.0  # a confident person
    logits[1, 3] = 5.0  # a confident car: dropped
    logits[2, 1] = -2.0  # a weak person: below the threshold
    boxes = np.array([[0.5, 0.5, 1.2, 0.5], [0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]], np.float32)
    masks = np.zeros((3, 4, 4), np.float32)
    masks[0, 1:3, 1:3] = 1.0
    out = _postprocess(boxes, logits, masks, 100, 50, 0.35)
    assert len(out) == 1
    box, mask = out[0]
    assert box == (0, 12, 100, 37)  # x clamped to the frame
    assert mask.shape == (50, 100) and mask[25, 50] and not mask[0, 0]
