"""``python -m streamsafe.prepare``: the installer / first-run step that builds or verifies the TensorRT engines,
and the system checker's "engines ready?" line. ONNX Runtime and the PC are faked (``tests/fake_ort.py``)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from streamsafe import prepare
from streamsafe import syscheck as sc
from streamsafe.config import Config
from streamsafe.models.onnx_backend import REQUIRED, engines, load_onnx_models, session, sha256
from tests import fake_ort
from tests.fake_ort import CPU, DML, ENGINE

ORDER = ["scene", "detector", "eye", "pose"]


@pytest.fixture
def models(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    files = {}
    for name in REQUIRED:
        (folder / name).write_bytes(name.encode())
        files[name] = {"sha256": sha256(folder / name), "bytes": len(name)}
    files["detector.onnx"]["model"] = "rfdetr RFDETRSegNano"
    (folder / "manifest.json").write_text(json.dumps({"files": files}))
    return folder


def run(models, *args) -> tuple[int, list[dict]]:
    events: list[dict] = []
    emit = lambda event, step=0, of=0, name="", detail="": events.append(dict(event=event, step=step, of=of, name=name, detail=detail))  # noqa: E731
    cfg = Config.from_env(env={}, onnx_dir=str(models), trt=True)
    return prepare.prepare(cfg, emit, check_only="--check" in args), events


def test_prepare_builds_every_engine_once_with_progress(models, monkeypatch, capsys):
    rt = fake_ort.install(monkeypatch)
    code, events = run(models, "--check")
    assert code == prepare.NOT_READY and rt.asked == [], "--check opens nothing and builds nothing"
    assert events[-1]["event"] == "todo" and events[-1]["detail"] == "scene, detector, eye, pose"

    code, events = run(models)
    assert code == prepare.READY and rt.built == 4
    steps = [(e["event"], e["step"], e["of"], e["name"]) for e in events if e["step"]]
    assert steps == [(ev, i, 4, name) for i, name in enumerate(ORDER, 1) for ev in ("preparing", "ready")]
    assert all(e["detail"].startswith("engine built in") for e in events if e["event"] == "ready")
    assert events[0]["event"] == "verifying" and events[-1]["event"] == "done"

    code, events = run(models)  # every later start: nothing to build, only the check
    assert code == prepare.READY and rt.built == 4 and rt.trt_asks() == 4, "verified engines are not even opened"
    assert [e["detail"] for e in events if e["event"] == "ready"] == ["verified"] * 4
    assert run(models, "--check")[0] == prepare.READY

    assert prepare.main(["--models", str(models)]) == 0  # the lines the app shows
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "ready 1/4: scene (verified)" and out[-1] == "PREPARE_OK 4/4 engines ready"


def test_the_engine_then_starts_from_the_prepared_engines_without_building(models, monkeypatch, tmp_path):
    from streamsafe.models.heads import FAST_FILE, MALE_FILE, THR_FILE

    rt = fake_ort.install(monkeypatch)
    assert run(models)[0] == prepare.READY
    for name, text in ((FAST_FILE, "100;0.5;10,20,30"), (MALE_FILE, "100;-0.5;1,2,3"), (THR_FILE, "0.2")):
        (tmp_path / name).write_text(text)
    monkeypatch.setattr("streamsafe.models.onnx_backend.scene.OnnxSigLIPScene.__init__", _scene_without_files)
    # the app's setting for a live start: never build now (TRT_BUILD=0). Everything must come from the cache
    cfg = Config.from_env(env={}, onnx_dir=str(models), trt=True, trt_build=False, head_dir=str(tmp_path))
    load_onnx_models(cfg)
    assert (rt.built, rt.loaded) == (4, 4) and session.device_summary() == "onnx-trt" and session.notes() == []

    # a setting that changes the engine (here the eye's input size) is not papered over: that model is not
    # loaded from an engine built for other settings
    load_onnx_models(replace(cfg, eye=replace(cfg.eye, eye_w=192)))
    assert (rt.built, rt.loaded) == (4, 7) and session.device_summary() == "onnx-trt+cuda"
    assert session.notes() == [f"eye.onnx: {session.OPENED[2].note}"] and "settings changed" in session.OPENED[2].note


def _scene_without_files(self, folder, head=None, allow_cpu=False, trt=None):
    from streamsafe.models.onnx_backend.session import open_session

    self.session = open_session(folder / "scene_vision.onnx", allow_cpu, trt)  # the stand-in folder has no real prompt files


def test_engine_start_without_prepare_runs_on_cuda_and_says_so(models, monkeypatch, tmp_path, capsys):
    from streamsafe.models.heads import FAST_FILE, MALE_FILE, THR_FILE

    rt = fake_ort.install(monkeypatch)
    for name, text in ((FAST_FILE, "100;0.5;10,20,30"), (MALE_FILE, "100;-0.5;1,2,3"), (THR_FILE, "0.2")):
        (tmp_path / name).write_text(text)
    monkeypatch.setattr("streamsafe.models.onnx_backend.scene.OnnxSigLIPScene.__init__", _scene_without_files)
    cfg = Config.from_env(env={}, onnx_dir=str(models), trt=True, trt_build=False, head_dir=str(tmp_path))
    load_onnx_models(cfg)
    assert rt.trt_asks() == 0 and session.device_summary() == "onnx-cuda" and len(session.notes()) == 4
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 4 and all(line.startswith("STREAMSAFE_DEVICE onnx-cuda | ") and "streamsafe.prepare" in line for line in err)


def test_a_new_driver_is_detected_rebuilt_and_the_old_engines_removed(models, monkeypatch):
    rt = fake_ort.install(monkeypatch)
    assert run(models)[0] == prepare.READY
    cache = models / "trt_cache"
    old = {p.name for p in cache.iterdir()}
    rt.identity = replace(rt.identity, gpus=((fake_ort.GPU[0][0], "566.14"),))

    code, events = run(models, "--check")
    assert code == prepare.NOT_READY and rt.built == 4
    assert [e["detail"] for e in events if e["event"] == "not_ready"] == ["the graphics driver changed (561.09 -> 566.14)"] * 4

    code, events = run(models)
    assert code == prepare.READY and rt.built == 8 and rt.loaded == 0, "rebuilt, the old engines never opened"
    assert "the graphics driver changed" in events[1]["detail"]
    assert events[-1]["detail"] == "removed 4 old engine folder(s)" and not old & {p.name for p in cache.iterdir()}


def test_one_damaged_engine_is_rebuilt_alone(models, monkeypatch):
    rt = fake_ort.install(monkeypatch)
    assert run(models)[0] == prepare.READY
    eye = next((models / "trt_cache").glob("eye-*"))
    (eye / ENGINE).write_bytes(b"engine ?")  # same size
    code, events = run(models, "--check")
    assert code == prepare.NOT_READY and events[-1]["detail"] == "eye" and "damaged" in events[3]["detail"]
    code, events = run(models)
    assert code == prepare.READY and rt.built == 5
    assert [e["name"] for e in events if e["event"] == "preparing"] == ["eye"]


@pytest.mark.parametrize(("mode", "reason"), [("fail", "out of memory"), ("nan", "non-finite"), ("skipped", "did not start")])
def test_failure_exits_non_zero_with_the_reason(models, monkeypatch, capsys, mode, reason):
    rt = fake_ort.install(monkeypatch)
    rt.trt = mode
    code = prepare.main(["--models", str(models), "--json"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == prepare.FAILED == 2 and lines[-1]["event"] == "failed" and lines[-1]["name"] == "scene"
    assert reason in lines[-1]["detail"] and rt.asked[-1][0][0] == fake_ort.TRT, "no CUDA session is opened in its place"
    assert run(models, "--check")[0] == prepare.NOT_READY


def test_failures_before_any_build_are_named(models, monkeypatch, tmp_path, capsys):
    fake_ort.install(monkeypatch, available=(DML, CPU))
    code, events = run(models)
    assert code == prepare.FAILED and "NVIDIA" in events[-1]["detail"]

    rt = fake_ort.install(monkeypatch)
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the engine folder should be")
    cfg = Config.from_env(env={}, onnx_dir=str(models), trt=True, trt_cache=str(blocked))
    events = []
    code = prepare.prepare(cfg, lambda event, step=0, of=0, name="", detail="": events.append((event, detail)))
    assert code == prepare.FAILED and "cannot be written" in events[-1][1] and rt.asked == []

    def no_identity(libs=""):
        raise engines.NotUsableError("the TensorRT libraries were not found")

    monkeypatch.setattr(engines, "identity", no_identity)
    assert prepare.main(["--models", str(models)]) == 2
    assert capsys.readouterr().out.splitlines()[-1] == "PREPARE_FAILED tensorrt: the TensorRT libraries were not found"

    (models / "pose.onnx").write_bytes(b"pose.onnY")  # a damaged model: nothing is built from it
    assert prepare.main(["--models", str(models)]) == 2
    assert capsys.readouterr().out.splitlines()[-1].startswith("PREPARE_FAILED models: ")
    monkeypatch.delenv("ONNX_DIR", raising=False)
    assert prepare.main([]) == 2 and "--models" in capsys.readouterr().out


def test_low_disk_space_is_a_warning_before_the_build(models, monkeypatch):
    fake_ort.install(monkeypatch)
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda p: type("U", (), {"free": 1.5e9})())
    code, events = run(models)
    assert code == prepare.READY and next(e["detail"] for e in events if e["event"] == "warning").startswith("1.5 GB free")


# ------------------------------------------------------------------------------- system checker: engines ready?
def test_syscheck_reports_engines_only_when_asked(models, monkeypatch):
    from tests.test_syscheck import by_name, probes

    fake_ort.install(monkeypatch)
    assert "engines" not in by_name(sc.run_checks(probes(), models=str(models), quick=True))
    check = by_name(sc.run_checks(probes(), models=str(models), quick=True, engines=True))["engines"]
    assert check.status == sc.WARN and "not ready: scene (not built yet)" in check.detail and check.fix_ar and check.fix_en

    assert run(models)[0] == prepare.READY
    for quick in (True, False):
        check = by_name(sc.run_checks(probes(), models=str(models), quick=quick, engines=True))["engines"]
        assert check.status == sc.OK and "ready (4/4)" in check.detail
    assert sc.verdict(sc.run_checks(probes(), models=str(models), engines=True)) == sc.OK

    def no_identity(libs=""):
        raise engines.NotUsableError("the TensorRT libraries were not found")

    monkeypatch.setattr(engines, "identity", no_identity)
    check = sc.check_engines(str(models), quick=True)
    assert check.status == sc.WARN and "libraries were not found" in check.detail, "a warning: CUDA still protects the stream"
    assert "slower mode" in check.fix_en and check.fix_ar
    assert sc.check_engines(str(models / "nope"), quick=True).status == sc.WARN
