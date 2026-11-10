"""System checker rules, with the machine faked."""

from __future__ import annotations

import json

from streamsafe import syscheck as sc
from streamsafe.syscheck import FAIL, OK, WARN, Gpu, Probes


def probes(**kw) -> Probes:
    base = dict(
        system=lambda: "Windows",
        python=lambda: (3, 11),
        ort_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        gpus=lambda: [Gpu("NVIDIA GeForce RTX 4060", "nvidia", 8.0, "561.09")],
        ram_gb=lambda: 16.0,
        disk_free_gb=lambda p: 100.0,
        which=lambda n: "C:/ss/ffmpeg.exe",
        exists=lambda p: True,
    )
    base.update(kw)
    return Probes(**base)


def by_name(checks):
    return {c.name: c for c in checks}


def test_good_nvidia_pc_passes():
    checks = sc.run_checks(probes())
    assert sc.verdict(checks) == OK, [c for c in checks if c.status != OK]


def test_no_gpu_runtime_fails_closed():
    c = by_name(sc.run_checks(probes(ort_providers=lambda: ["CPUExecutionProvider"])))
    assert c["runtime"].status == FAIL and c["runtime"].fix_ar


def test_amd_on_directml_is_ok():
    c = by_name(
        sc.run_checks(
            probes(
                ort_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
                gpus=lambda: [Gpu("AMD Radeon RX 7600", "amd", None)],
            )
        )
    )
    assert c["runtime"].status == OK and c["gpu"].status == OK


def test_nvidia_on_directml_suggests_nvidia_edition():
    c = by_name(sc.run_checks(probes(ort_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])))
    assert c["runtime"].status == WARN


def test_no_gpu_and_low_vram():
    assert by_name(sc.run_checks(probes(gpus=list)))["gpu"].status == FAIL
    low = by_name(sc.run_checks(probes(gpus=lambda: [Gpu("NVIDIA GeForce GTX 1650", "nvidia", 4.0)])))
    assert low["gpu"].status == WARN


def test_discrete_card_preferred_over_integrated():
    gpus = [Gpu("Intel(R) UHD Graphics 770", "intel", None), Gpu("NVIDIA GeForce RTX 3060", "nvidia", 12.0)]
    _, best = sc.check_gpu(probes(gpus=lambda: gpus))
    assert best is not None and best.vendor == "nvidia"


def test_missing_ffmpeg_and_old_python_fail():
    c = by_name(sc.run_checks(probes(which=lambda n: None, python=lambda: (3, 10))))
    assert c["ffmpeg"].status == FAIL and c["python"].status == FAIL


def test_vendor_names():
    assert sc.vendor_of("NVIDIA GeForce RTX 4090") == "nvidia"
    assert sc.vendor_of("AMD Radeon RX 7900 XTX") == "amd"
    assert sc.vendor_of("Intel(R) Arc(TM) A770 Graphics") == "intel"


def test_models_folder(tmp_path):
    from streamsafe.models.onnx_backend import REQUIRED, sha256

    files = {}
    for name in REQUIRED:
        (tmp_path / name).write_bytes(name.encode())
        files[name] = {"bytes": len(name.encode()), "sha256": sha256(tmp_path / name)}
    (tmp_path / "manifest.json").write_text(json.dumps({"files": files}))
    assert sc.check_models(str(tmp_path), quick=False).status == OK
    (tmp_path / "pose.onnx").write_bytes(b"xx")  # wrong size
    assert sc.check_models(str(tmp_path), quick=True).status == FAIL
    assert sc.check_models(str(tmp_path / "nope"), quick=True).status == FAIL


def test_main_json(capsys, monkeypatch):
    monkeypatch.setattr(sc, "Probes", lambda: probes())
    code = sc.main(["--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["verdict"] == OK
