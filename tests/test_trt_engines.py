"""The TensorRT engine cache and the TensorRT -> CUDA -> refuse chain, with ONNX Runtime and the PC faked
(``tests/fake_ort.py``). What must hold: a stale, partial or damaged engine is never loaded; a model that
TensorRT cannot run runs on CUDA and the reason is kept; nothing ever runs on the CPU."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from streamsafe.models.onnx_backend import engines, session
from streamsafe.models.onnx_backend.session import EngineError, Trt, open_session
from tests import fake_ort
from tests.fake_ort import CPU, CUDA, DML, ENGINE, TRT

FILES = {"eye.onnx": {"sha256": "e1"}, "eye.onnx.data": {"sha256": "d1"}, "eye16.onnx": {"sha256": "h1"}, "pose.onnx": {"sha256": "p1"}}
IDENT = engines.Identity(fake_ort.GPU, "1.22.0", "tensorrt-cu12-libs 10.9.0.34")
SHAPES = {"pixels": ((1, 3, 8, 8), (4, 3, 8, 8), (16, 3, 8, 8))}


def trt(tmp_path, **kw) -> Trt:
    return Trt(**{**dict(cache=str(tmp_path / "trt"), fp16=True, files=FILES), **kw})


def key(model="eye.onnx", files=FILES, ident=IDENT, fp16=True, shapes=SHAPES) -> dict:
    options = session.trt_options(Trt("x", fp16), shapes)
    return engines.engine_key(model, engines.model_shas(Path(model), files), options, ident)


def built(tmp_path, k: dict | None = None, model="eye.onnx", content=b"engine bytes") -> Path:
    """A complete engine folder made the way a build makes it."""
    k = k or key(model)
    final = engines.folder(tmp_path / "trt", model, k)
    work = engines.begin(final)
    (work / ENGINE).write_bytes(content)
    assert engines.commit(work, final, k) == ""
    return final


# ------------------------------------------------------------------------------------------- the key
def test_key_follows_everything_an_engine_depends_on(tmp_path):
    base = engines.folder(tmp_path, "eye.onnx", key())
    assert base.name.startswith("eye-") and base == engines.folder(tmp_path, "eye.onnx", key()), "same PC, same model: same folder"
    gpu, driver = fake_ort.GPU[0]
    others = dict(
        model_bytes=key(files={**FILES, "eye.onnx": {"sha256": "e2"}}),
        weight_bytes=key(files={**FILES, "eye.onnx.data": {"sha256": "d2"}}),
        card=key(ident=replace(IDENT, gpus=(("NVIDIA GeForce RTX 4060", driver),))),
        driver=key(ident=replace(IDENT, gpus=((gpu, "566.14"),))),
        second_card=key(ident=replace(IDENT, gpus=((gpu, driver), ("NVIDIA GeForce GTX 1650", driver)))),
        onnx_runtime=key(ident=replace(IDENT, ort="1.23.0")),
        tensorrt=key(ident=replace(IDENT, trt="tensorrt-cu12-libs 10.13.0.35")),
        fp32=key(fp16=False),
        batch_range={**key(shapes={"pixels": ((1, 3, 8, 8), (4, 3, 8, 8), (8, 3, 8, 8))})},
    )
    names = {what: engines.folder(tmp_path, "eye.onnx", k).name for what, k in others.items()}
    assert base.name not in names.values() and len(set(names.values())) == len(names), names
    assert engines.folder(tmp_path, "eye16.onnx", key("eye16.onnx")).name.startswith("eye16-")


def test_a_model_outside_a_manifest_is_hashed_so_the_key_still_follows_its_bytes(tmp_path):
    model = tmp_path / "loose.onnx"
    model.write_bytes(b"one")
    first = engines.model_shas(model, {})
    model.write_bytes(b"two")
    assert first and set(first) == {"loose.onnx"} and engines.model_shas(model, {}) != first
    assert engines.model_shas(Path("eye.onnx"), FILES) == {"eye.onnx": "e1", "eye.onnx.data": "d1"}, "the weight file counts"


def test_identity_needs_the_card_the_driver_and_the_libraries(tmp_path, monkeypatch):
    from streamsafe import syscheck

    monkeypatch.setattr(syscheck, "nvidia_gpus", list)
    with pytest.raises(engines.NotUsableError, match="nvidia-smi"):
        engines.identity()
    monkeypatch.setattr(syscheck, "nvidia_gpus", lambda: [syscheck.Gpu("NVIDIA GeForce RTX 3090", "nvidia", 24.0, "561.09")])
    with pytest.raises(engines.NotUsableError, match="TensorRT libraries"):
        engines.identity(str(tmp_path))  # an empty TRT_LIBS folder
    (tmp_path / "nvinfer_plugin_10.dll").write_bytes(b"plugin")  # not the core library
    (tmp_path / "nvinfer_10.dll").write_bytes(b"core library")
    ident = engines.identity(str(tmp_path))
    assert ident.gpus == (("NVIDIA GeForce RTX 3090", "561.09"),) and ident.trt == "nvinfer_10.dll 12 bytes"
    (tmp_path / "nvinfer_10.dll").write_bytes(b"core library of another release")
    assert engines.identity(str(tmp_path)).trt != ident.trt, "other libraries, other key"


# ------------------------------------------------------------------------------- complete, or not loaded
def test_a_sealed_engine_passes_and_an_interrupted_build_is_never_trusted(tmp_path):
    k = key()
    final = engines.folder(tmp_path / "trt", "eye.onnx", k)
    assert engines.check(final, k) == "not built yet"
    work = engines.begin(final)
    (work / ENGINE).write_bytes(b"half an eng")  # the PC is switched off here: no seal, nothing moved into place
    assert not final.exists() and engines.check(final, k) == "not built yet"
    assert engines.why_not_ready(tmp_path / "trt", "eye.onnx", k) == "not built yet", "a .building folder is nobody's engine"

    final = built(tmp_path, k)
    assert engines.check(final, k) == "" and engines.check(final, k, quick=True) == ""
    assert list((tmp_path / "trt").glob("*.building-*")) == [work], "the interrupted build is still there, unread"
    assert json.loads((final / engines.SEAL).read_text())["key"] == k, "the full key is written next to the engine"

    (final / engines.SEAL).unlink()  # engine files without a seal: as good as an interrupted build
    assert "no seal" in engines.check(final, k)


@pytest.mark.parametrize(
    ("damage", "expect", "quick_sees_it"),
    [
        (lambda f: (f / ENGINE).write_bytes(b"engine byt"), "damaged", True),  # cut short
        (lambda f: (f / ENGINE).write_bytes(b"engine bytEs"), "damaged", False),  # same size, other bytes
        (lambda f: (f / ENGINE).unlink(), "does not hold the files", True),
        (lambda f: (f / "extra.engine").write_bytes(b"x"), "does not hold the files", True),  # ONNX Runtime could pick it up
        (lambda f: (f / engines.SEAL).write_text("{not json"), "unreadable", True),
        (lambda f: (f / engines.SEAL).write_text(json.dumps({"key": key()})), "unreadable", True),
        (lambda f: (f / engines.SEAL).write_text(json.dumps({"key": key(), "files": {ENGINE: {"bytes": "many"}}})), "unreadable", True),
        (lambda f: (f / engines.SEAL).write_text(json.dumps({"key": key(), "files": [ENGINE]})), "unreadable", True),
    ],
)
def test_a_damaged_engine_folder_is_refused(tmp_path, damage, expect, quick_sees_it):
    final = built(tmp_path)
    damage(final)
    assert expect in engines.check(final, key())
    assert bool(engines.check(final, key(), quick=True)) == quick_sees_it, "the quick check compares sizes, the engine always hashes"


def test_a_build_that_left_no_engine_is_not_stored(tmp_path):
    final = engines.folder(tmp_path / "trt", "eye.onnx", key())
    work = engines.begin(final)
    (work / "TensorrtExecutionProvider_cache.timing").write_bytes(b"timings only")
    assert "no engine file" in engines.commit(work, final, key())
    assert not final.exists() and not work.exists()


def test_a_seal_made_for_another_pc_is_refused_and_the_reason_names_the_change(tmp_path):
    gpu, driver = fake_ort.GPU[0]
    old = built(tmp_path)
    cases = {
        "the graphics driver changed (561.09 -> 566.14)": key(ident=replace(IDENT, gpus=((gpu, "566.14"),))),
        "the graphics card changed (NVIDIA GeForce RTX 3090 -> NVIDIA GeForce RTX 4060)": key(
            ident=replace(IDENT, gpus=(("NVIDIA GeForce RTX 4060", driver),))
        ),
        "TensorRT changed (tensorrt-cu12-libs 10.9.0.34 -> tensorrt-cu12-libs 10.13.0.35)": key(
            ident=replace(IDENT, trt="tensorrt-cu12-libs 10.13.0.35")
        ),
        "ONNX Runtime changed (1.22.0 -> 1.23.0)": key(ident=replace(IDENT, ort="1.23.0")),
        "the model file changed": key(files={**FILES, "eye.onnx.data": {"sha256": "d2"}}),
        "the engine settings changed": key(fp16=False),
    }
    for reason, new in cases.items():
        assert engines.check(old, new) == reason, "even under the right folder name the seal's key decides"
        assert engines.folder(tmp_path / "trt", "eye.onnx", new) != old
        assert engines.why_not_ready(tmp_path / "trt", "eye.onnx", new) == reason, "what the streamer is told"
    assert engines.why_not_ready(tmp_path / "trt", "pose.onnx", key("pose.onnx")) == "not built yet"


def test_two_builds_of_one_engine_do_not_disturb_each_other(tmp_path):
    k = key()
    final = engines.folder(tmp_path / "trt", "eye.onnx", k)
    first, second = engines.begin(final), engines.begin(final)
    assert first != second
    (first / ENGINE).write_bytes(b"first")
    (second / ENGINE).write_bytes(b"second")
    assert engines.commit(first, final, k) == "" and engines.commit(second, final, k) == ""
    assert (final / ENGINE).read_bytes() == b"first" and engines.check(final, k) == "", "the one already in place is kept whole"
    assert not first.exists() and not second.exists()

    (final / ENGINE).write_bytes(b"dam")  # a damaged one in place is replaced by the next build
    third = engines.begin(final)
    (third / ENGINE).write_bytes(b"third")
    assert engines.commit(third, final, k) == "" and (final / ENGINE).read_bytes() == b"third" and engines.check(final, k) == ""


def test_sweep_removes_old_engines_and_killed_builds_only(tmp_path):
    gpu, _ = fake_ort.GPU[0]
    cache = tmp_path / "trt"
    old = built(tmp_path)
    current = built(tmp_path, key(ident=replace(IDENT, gpus=((gpu, "566.14"),))))
    other_model = built(tmp_path, key("eye16.onnx"), "eye16.onnx")
    killed, running = engines.begin(current), engines.begin(current)
    os.utime(killed, (time.time() - 2 * engines.STALE_BUILD_S,) * 2)
    (cache / "eye-notes").mkdir()  # not a name this module gives: never touched
    (cache / "eye-notes" / "keep.txt").write_text("mine")
    removed = engines.sweep(cache, [current])
    assert set(removed) == {old.name, killed.name}
    assert current.exists() and other_model.exists() and running.exists() and (cache / "eye-notes" / "keep.txt").exists()
    engines.remove(cache / "eye-notes")
    assert (cache / "eye-notes").exists(), "remove() only deletes engine folders"


# ------------------------------------------------------------------- the chain: TensorRT -> CUDA -> refuse
def test_first_start_builds_and_seals_later_starts_load(tmp_path, monkeypatch):
    rt = fake_ort.install(monkeypatch)
    first = open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    assert first.get_providers()[0] == TRT and rt.built == 1 and rt.loaded == 0
    assert rt.asked[0][1:] == [CUDA], "what TensorRT cannot run stays on CUDA; the CPU is never asked for"
    options = rt.asked[0][0][1]
    assert options["trt_fp16_enable"] is True and options["trt_engine_cache_enable"] is True
    assert options["trt_profile_min_shapes"] == "pixels:1x3x8x8" and options["trt_profile_max_shapes"] == "pixels:16x3x8x8"
    assert engines.BUILDING in options["trt_engine_cache_path"], "a build never writes into the folder engines are loaded from"
    final = engines.folder(tmp_path / "trt", "eye.onnx", key())
    assert engines.check(final, key()) == "" and not list((tmp_path / "trt").glob("*.building-*"))
    assert session.OPENED[-1].note.startswith("engine built in") and session.device_summary() == "onnx-trt"

    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    assert (rt.built, rt.loaded) == (1, 1) and rt.asked[-1][0][1]["trt_engine_cache_path"] == str(final)
    assert session.OPENED[-1].note == "" and session.notes() == [f"eye.onnx: {session.OPENED[0].note}"]


@pytest.mark.parametrize("what", ["driver", "card", "tensorrt", "onnxruntime", "model"])
def test_a_change_of_gpu_driver_or_libraries_rebuilds_and_never_loads_the_old_engine(tmp_path, monkeypatch, what):
    rt = fake_ort.install(monkeypatch)
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    old = engines.folder(tmp_path / "trt", "eye.onnx", key())
    gpu, driver = fake_ort.GPU[0]
    files = FILES
    if what == "driver":
        rt.identity = replace(IDENT, gpus=((gpu, "566.14"),))
    elif what == "card":
        rt.identity = replace(IDENT, gpus=(("NVIDIA GeForce RTX 4060", driver),))
    elif what == "tensorrt":
        rt.identity = replace(IDENT, trt="tensorrt-cu12-libs 10.13.0.35")
    elif what == "onnxruntime":
        rt.identity = replace(IDENT, ort="1.23.0")
    else:
        files = {**FILES, "eye.onnx.data": {"sha256": "d2"}}

    # the app said "never build at start": the model runs on CUDA, the old engine is not even opened
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path, files=files, build=False), shapes=SHAPES)
    assert rt.trt_asks() == 1 and rt.loaded == 0 and session.OPENED[-1].device == "cuda"
    assert "changed" in session.OPENED[-1].note and "streamsafe.prepare" in session.OPENED[-1].note

    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path, files=files), shapes=SHAPES)  # building allowed: a new engine
    assert (rt.built, rt.loaded) == (2, 0) and session.OPENED[-1].device == "trt" and "changed" in session.OPENED[-1].note
    assert str(old) not in [p[0][1]["trt_engine_cache_path"] for p in rt.asked if not isinstance(p[0], str)][1:]


def test_a_damaged_engine_is_never_loaded(tmp_path, monkeypatch):
    rt = fake_ort.install(monkeypatch)
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    final = engines.folder(tmp_path / "trt", "eye.onnx", key())
    (final / ENGINE).write_bytes(b"engine X")  # same size as "engine 1", other bytes
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path, build=False), shapes=SHAPES)
    assert rt.trt_asks() == 1 and rt.loaded == 0, "TensorRT was not even asked to open it"
    assert session.OPENED[-1].device == "cuda" and "damaged" in session.OPENED[-1].note
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    assert (rt.built, rt.loaded) == (2, 0) and engines.check(final, key()) == "" and (final / ENGINE).read_bytes() == b"engine 2"


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("fail", "out of memory"),
        ("fail_cpu", "out of memory"),  # ONNX Runtime 1.22.0 retried on the CPU by itself: that session is not used
        ("skipped", "did not start"),
        ("nan", "non-finite"),
        ("noengine", "no engine file"),  # the provider started but compiled nothing: that is CUDA under another name
    ],
)
def test_tensorrt_failure_falls_back_to_cuda_with_the_reason(tmp_path, monkeypatch, mode, reason):
    rt = fake_ort.install(monkeypatch)
    rt.trt = mode
    s = open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    opened = session.OPENED[-1]
    assert s.get_providers()[0] == CUDA and opened.device == "cuda" and rt.asked[-1] == [CUDA, CPU]
    assert opened.note.startswith("TensorRT not used (") and reason in opened.note and opened.note.endswith("running on CUDA")
    assert session.device_summary() == "onnx-cuda"
    cache = tmp_path / "trt"
    assert not [p for p in cache.iterdir() if p.is_dir()], "nothing sealed, nothing half-built left behind"
    with pytest.raises(EngineError, match=reason):  # the prepare step: an error, never a quiet CUDA session
        open_session(tmp_path / "eye.onnx", trt=trt(tmp_path, strict=True), shapes=SHAPES)


def test_an_unknown_gpu_or_missing_libraries_mean_cuda_not_a_blind_engine(tmp_path, monkeypatch):
    rt = fake_ort.install(monkeypatch)

    def no_identity(libs=""):
        raise engines.NotUsableError("the NVIDIA card and its driver version could not be read (nvidia-smi)")

    monkeypatch.setattr(engines, "identity", no_identity)
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    assert rt.trt_asks() == 0 and session.OPENED[-1].device == "cuda" and "nvidia-smi" in session.OPENED[-1].note
    assert not (tmp_path / "trt").exists()


def test_a_stored_engine_that_does_not_load_is_deleted_and_built_again(tmp_path, monkeypatch):
    rt = fake_ort.install(monkeypatch)
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    final = engines.folder(tmp_path / "trt", "eye.onnx", key())
    rt.trt = "fail"  # e.g. TensorRT refuses to read it
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path, build=False), shapes=SHAPES)
    assert session.OPENED[-1].device == "cuda" and "did not load" in session.OPENED[-1].note
    assert not final.exists(), "it is not offered again: the next prepare builds a new one"
    rt.trt = "ok"
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    assert session.OPENED[-1].device == "trt" and engines.check(final, key()) == ""


def test_an_engine_onnx_runtime_did_not_take_is_resealed(tmp_path, monkeypatch):
    rt = fake_ort.install(monkeypatch)
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    final = engines.folder(tmp_path / "trt", "eye.onnx", key())
    rt.trt = "ignore"  # ONNX Runtime looks for another file name and builds its own at load
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    assert session.OPENED[-1].device == "trt" and "not used" in session.OPENED[-1].note, "a slow start is not hidden"
    assert engines.check(final, key()) == "", "the folder matches its seal again"


def test_one_model_off_tensorrt_shows_in_the_device_name(tmp_path, monkeypatch):
    from streamsafe.runtime import report

    rt = fake_ort.install(monkeypatch)
    open_session(tmp_path / "pose.onnx", trt=trt(tmp_path), shapes=SHAPES)
    rt.trt = "fail"
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    assert report.device_name() == "onnx-trt+cuda"
    notes = report.device_notes()
    assert len(notes) == 2 and notes[1].startswith("eye.onnx: TensorRT not used (the engine could not be built (")
    stats: dict = {"headroom": [{"budget": 0}]}
    report.device_stats(stats)
    summary = report.device_summary(stats)
    assert summary == dict(gpu="onnx-trt+cuda", device_notes=notes, headroom={"budget": 0}) and not stats
    assert report.device_summary({}) == dict(gpu="?", device_notes=[], headroom=None), "a worker that died reports nothing"


def test_tensorrt_being_installed_changes_nothing_by_itself(tmp_path, monkeypatch):
    rt = fake_ort.install(monkeypatch)
    plain = open_session(tmp_path / "eye.onnx")
    assert rt.asked == [[CUDA, CPU]] and plain.get_providers()[0] == CUDA
    assert session.device_summary() == "onnx-cuda" and session.notes() == []


def test_no_cuda_behind_tensorrt_refuses_to_start(tmp_path, monkeypatch):
    fake_ort.install(monkeypatch, available=(TRT, DML, CPU))
    with pytest.raises(RuntimeError, match="TRT=1"):
        open_session(tmp_path / "eye.onnx", trt=trt(tmp_path))


def test_tensorrt_and_cuda_both_failing_refuses_to_start_never_the_cpu(tmp_path, monkeypatch):
    rt = fake_ort.install(monkeypatch)
    rt.trt, rt.cuda = "fail", False
    with pytest.raises(RuntimeError, match="could not start on the GPU"):
        open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    rt.trt = "skipped"  # TensorRT and CUDA both absent at run time: ONNX Runtime would quietly use the CPU
    with pytest.raises(RuntimeError, match="could not start on the GPU"):
        open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    assert not session.OPENED


def test_provider_without_tensorrt_in_onnx_runtime(tmp_path, monkeypatch):
    rt = fake_ort.install(monkeypatch, available=(CUDA, CPU))
    open_session(tmp_path / "eye.onnx", trt=trt(tmp_path), shapes=SHAPES)
    assert rt.trt_asks() == 0 and session.OPENED[-1].device == "cuda" and "no TensorRT provider" in session.OPENED[-1].note


def test_the_real_onnx_runtime_accepts_how_tensorrt_sessions_are_opened(tmp_path):
    """No TensorRT here, so the real session comes back on the CPU: it must be refused with a reason, and the
    subclass and arguments used (``enable_fallback``, the provider entry) must be ones ONNX Runtime takes."""
    pytest.importorskip("onnx")
    import warnings

    import onnxruntime as ort
    from onnx import helper

    from tests.test_onnx_backend import _save

    node = helper.make_node("ReduceMean", ["pixels"], ["summary"], axes=[2, 3], keepdims=0)
    path = _save(tmp_path, "eye.onnx", [node], ("pixels", ["n", 3, 8, 8]), ("summary", ["n", 3]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got, err = session._trt_session(ort, path, ort.SessionOptions(), session.trt_options(trt(tmp_path), SHAPES), tmp_path, SHAPES)
    assert got is None and "did not start" in err
    real = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session._selftest(real, SHAPES) == "" and session._selftest(real, None) == "", "the test input fits a real model"
