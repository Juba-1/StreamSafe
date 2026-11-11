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


# ------------------------------------------------------------------------------------- TensorRT
# (the engine cache and the TensorRT -> CUDA -> refuse chain: tests/test_trt_engines.py)
def test_trt_options_are_what_the_engine_is_built_with(tmp_path):
    from streamsafe.models.onnx_backend.session import Trt, trt_options

    shapes = {"pixels": ((1, 3, 384, 224), (4, 3, 384, 224), (16, 3, 384, 224))}
    opt = trt_options(Trt(str(tmp_path)), shapes)
    assert opt["trt_fp16_enable"] is True and opt["trt_layer_norm_fp32_fallback"] is True
    assert opt["trt_profile_min_shapes"] == "pixels:1x3x384x224"
    assert opt["trt_profile_opt_shapes"] == "pixels:4x3x384x224"
    assert opt["trt_profile_max_shapes"] == "pixels:16x3x384x224"
    fixed = trt_options(Trt(str(tmp_path), fp16=False))
    assert fixed["trt_fp16_enable"] is False and not any("profile" in k for k in fixed)
    assert not any("path" in k for k in opt), "folders are not part of what an engine is: the key must not move with them"


def test_the_engine_and_the_prepare_step_ask_for_the_same_engines():
    """One list of models and batch ranges (``trt_models``) for both, and no setting a streamer changes in it."""
    from streamsafe.config import Config
    from streamsafe.models.onnx_backend import eye, pose, trt_models

    models = trt_models(Config.from_env(env={}))
    assert [(label, file) for label, file, _ in models] == [
        ("scene", "scene_vision.onnx"),
        ("detector", "detector.onnx"),
        ("eye", "eye.onnx"),
        ("pose", "pose.onnx"),
    ]
    assert models[2][2] == eye.shapes(224, 384) == {"pixels": ((1, 3, 384, 224), (4, 3, 384, 224), (16, 3, 384, 224))}
    assert models[3][2] is pose.SHAPES and pose.SHAPES["pixels"][2][0] == pose.MAX_BATCH
    budget = trt_models(Config.from_env(env={}, eye_budget=6, eye_fp16=True, scene_prior=False, scene_guard=False))
    assert [file for _, file, _ in budget] == ["detector.onnx", "eye16.onnx", "pose.onnx"]
    assert budget[1][2] == models[2][2], "an eye budget must not change the engine (a rebuild of minutes)"


def test_detector_nan_fails_the_decision_instead_of_finding_nobody():
    logits = np.full((2, 91), -10.0, np.float32)
    logits[0, 1] = np.nan
    boxes = np.full((2, 4), 0.5, np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        _postprocess(boxes, logits, np.zeros((2, 4, 4), np.float32), 100, 50, 0.35)


# ------------------------------------------------------------------- the model classes on real ONNX Runtime (CPU)
def _save(tmp_path, name, nodes, inp, out, init=()):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(inp[0], TensorProto.FLOAT, inp[1])],
        [helper.make_tensor_value_info(out[0], TensorProto.FLOAT, out[1])],
        list(init),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(tmp_path / name))
    return tmp_path / name


def test_eye_runs_long_batches_in_parts_with_the_same_result(tmp_path):
    pytest.importorskip("onnx")
    from onnx import helper

    from streamsafe.models.heads import FAST_FILE, MALE_FILE, THR_FILE
    from streamsafe.models.onnx_backend.eye import MAX_BATCH, OnnxRadioEye

    node = helper.make_node("ReduceMean", ["pixels"], ["summary"], axes=[2, 3], keepdims=0)
    path = _save(tmp_path, "eye.onnx", [node], ("pixels", ["n", 3, 48, 32]), ("summary", ["n", 3]))
    (tmp_path / FAST_FILE).write_text("100;0.5;10,20,30")
    (tmp_path / MALE_FILE).write_text("100;-0.5;1,2,3")
    (tmp_path / THR_FILE).write_text("0.2")
    eye = OnnxRadioEye(path, 32, 48, str(tmp_path), allow_cpu=True)
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (60, 30, 3), np.uint8) for _ in range(MAX_BATCH + 4)]
    emb = eye.embed(crops)
    assert emb.shape == (MAX_BATCH + 4, 3)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert np.allclose(emb[-1], eye.embed(crops[-1:])[0], atol=1e-6), "a crop scores the same alone and in a crowd"


def test_pose_runs_long_batches_in_parts(tmp_path):
    pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    from streamsafe.models.onnx_backend.pose import MAX_BATCH, OnnxViTPose

    w = numpy_helper.from_array(np.random.default_rng(0).normal(size=(17, 3, 1, 1)).astype(np.float32), "w")
    nodes = [
        helper.make_node("AveragePool", ["pixels"], ["small"], kernel_shape=[4, 4], strides=[4, 4]),
        helper.make_node("Conv", ["small", "w"], ["heatmaps"]),
    ]
    path = _save(tmp_path, "pose.onnx", nodes, ("pixels", ["n", 3, 256, 192]), ("heatmaps", ["n", 17, 64, 48]), [w])
    pose = OnnxViTPose(path, allow_cpu=True)
    rgb = np.random.default_rng(1).integers(0, 255, (540, 960, 3), np.uint8)
    boxes = [(10 + 20 * i, 50, 70 + 20 * i, 300) for i in range(MAX_BATCH + 3)]
    out = pose(rgb, boxes)
    assert len(out) == MAX_BATCH + 3 and out[0][0].shape == (17, 2)
    alone = pose(rgb, boxes[-1:])[0]
    assert np.allclose(out[-1][0], alone[0], atol=1e-3) and np.allclose(out[-1][1], alone[1], atol=1e-5)


def test_speed_bench_measures_every_batch_size(tmp_path):
    pytest.importorskip("onnx")
    from onnx import helper

    from bench.onnx_speed import _first_input, measure
    from streamsafe.config import Config

    node = helper.make_node("ReduceMean", ["pixels"], ["summary"], axes=[2, 3], keepdims=0)
    path = _save(tmp_path, "eye.onnx", [node], ("pixels", ["n", 3, 48, 32]), ("summary", ["n", 3]))
    assert _first_input(path) == ("pixels", ["n", 3, 48, 32])
    out = measure(path, [1, 2, 4], runs=4, cfg=Config.from_env(env={}, allow_cpu=True))
    assert out["device"] == "cpu" and set(out["batch"]) == {1, 2, 4}
    assert all({"first_ms", "p50", "p95", "per_item"} <= set(row) for row in out["batch"].values())
    assert "switch_extra_ms" in out and "bound_p50" not in out["batch"][1], "I/O binding is a GPU measurement"
