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
def _trt(tmp_path, data_sha="d1", fp16=True):
    from streamsafe.models.onnx_backend.session import Trt

    files = {"eye.onnx": {"sha256": "e1"}, "eye.onnx.data": {"sha256": data_sha}, "eye16.onnx": {"sha256": "h1"}}
    return Trt(str(tmp_path / "trt"), fp16, files)


def test_trt_engine_folder_follows_the_model_bytes(tmp_path):
    """ONNX Runtime finds a cached engine by name only: other weights under the same name must not find it."""
    name = _trt(tmp_path).folder("eye.onnx").name
    assert name.startswith("eye-") and name.endswith("-fp16")
    assert _trt(tmp_path, data_sha="d2").folder("eye.onnx").name != name, "new weight file, new engine"
    assert _trt(tmp_path, fp16=False).folder("eye.onnx").name != name
    assert _trt(tmp_path).folder("eye16.onnx").name != name
    assert _trt(tmp_path).folder("eye.onnx").name == name


def test_trt_provider_options(tmp_path):
    from streamsafe.models.onnx_backend.session import trt_provider

    shapes = {"pixels": ((1, 3, 384, 224), (4, 3, 384, 224), (16, 3, 384, 224))}
    name, opt = trt_provider(_trt(tmp_path), "eye.onnx", shapes)
    assert name == "TensorrtExecutionProvider"
    assert opt["trt_fp16_enable"] is True and opt["trt_engine_cache_enable"] is True
    assert (tmp_path / "trt").is_dir() and opt["trt_engine_cache_path"].startswith(str(tmp_path / "trt"))
    assert opt["trt_profile_min_shapes"] == "pixels:1x3x384x224"
    assert opt["trt_profile_opt_shapes"] == "pixels:4x3x384x224"
    assert opt["trt_profile_max_shapes"] == "pixels:16x3x384x224"
    _, fixed = trt_provider(_trt(tmp_path, fp16=False), "detector.onnx")
    assert fixed["trt_fp16_enable"] is False and not any("profile" in k for k in fixed)


class _FakeSession:
    def __init__(self, path, sess_options, providers):
        self.asked = providers

    def get_providers(self):
        return [p if isinstance(p, str) else p[0] for p in self.asked]


def test_trt_goes_in_front_of_cuda_only_when_asked(tmp_path, monkeypatch):
    import onnxruntime as ort

    from streamsafe.models.onnx_backend import session

    names = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(ort, "get_available_providers", lambda: names)
    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)
    monkeypatch.delenv("STREAMSAFE_EP", raising=False)
    plain = session.open_session(tmp_path / "eye.onnx")
    assert plain.asked == names[1:], "TensorRT being installed changes nothing by itself"
    assert session.LAST_DEVICE == "onnx-cuda"
    fast = session.open_session(tmp_path / "eye.onnx", trt=_trt(tmp_path), shapes={"pixels": ((1, 3, 8, 8),) * 3})
    assert fast.asked[0][0] == names[0] and fast.asked[1:] == names[1:], "what TensorRT cannot run stays on CUDA"
    assert session.LAST_DEVICE == "onnx-trt"


def test_trt_without_nvidia_stops_at_start(tmp_path, monkeypatch):
    import onnxruntime as ort

    from streamsafe.models.onnx_backend import session

    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)
    monkeypatch.delenv("STREAMSAFE_EP", raising=False)
    for names in (["CUDAExecutionProvider", "CPUExecutionProvider"], ["TensorrtExecutionProvider", "DmlExecutionProvider"]):
        monkeypatch.setattr(ort, "get_available_providers", lambda names=names: names)
        with pytest.raises(RuntimeError, match="TRT=1"):
            session.open_session(tmp_path / "eye.onnx", trt=_trt(tmp_path))


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
