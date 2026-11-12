"""Downloads: https only, resume, sha256, the model files from a manifest, atomic install, clear errors."""

import json
import urllib.request

import pytest

from streamsafe.app import fetch, modelpack
from tests.appfix import Server, light_verify, make_pack, sha

REAL_VERIFY = modelpack.verify  # before the stand-in below replaces it


@pytest.fixture(autouse=True)
def light(monkeypatch):
    monkeypatch.setattr(modelpack, "verify", light_verify)


def test_https_only_and_pinned_sha(tmp_path):
    with pytest.raises(fetch.DownloadError, match="https only"):
        fetch.download("http://example.com/x", tmp_path / "x", "0" * 64)
    with pytest.raises(fetch.DownloadError, match="https only"):
        fetch.download("ftp://example.com/x", tmp_path / "x", "0" * 64)
    with pytest.raises(fetch.IntegrityError, match="no sha256"):
        fetch.download("https://example.com/x", tmp_path / "x", "")


def test_redirect_to_http_is_refused(tmp_path):
    handler = fetch._HttpsOnlyRedirects()
    req = urllib.request.Request("https://huggingface.co/x")
    with pytest.raises(fetch.DownloadError, match="https only"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://evil.example/x")


def test_download_ok_wrong_sha_deleted_and_status_kept(tmp_path):
    src = tmp_path / "srv"
    src.mkdir()
    (src / "a.bin").write_bytes(b"hello" * 1000)
    with Server(src) as s:
        out = fetch.download(s.url + "a.bin", tmp_path / "d" / "a.bin", sha(b"hello" * 1000), allow_http=True)
        assert out.read_bytes() == b"hello" * 1000
        with pytest.raises(fetch.IntegrityError):
            fetch.download(s.url + "a.bin", tmp_path / "d" / "b.bin", sha(b"other"), allow_http=True)
        assert not (tmp_path / "d" / "b.bin").exists() and not (tmp_path / "d" / "b.bin.part").exists()
        with pytest.raises(fetch.DownloadError) as e:
            fetch.download(s.url + "missing.bin", tmp_path / "d" / "c.bin", sha(b"x"), allow_http=True)
        assert e.value.status == 404


def test_part_file_is_continued_or_restarted(tmp_path):
    src = tmp_path / "srv"
    src.mkdir()
    data = bytes(range(256)) * 400
    (src / "a.bin").write_bytes(data)
    dest = tmp_path / "d" / "a.bin"
    dest.parent.mkdir()
    (dest.parent / "a.bin.part").write_bytes(data[:1000])  # a broken earlier try (the test server has no Range: restarts)
    with Server(src) as s:
        assert fetch.download(s.url + "a.bin", dest, sha(data), len(data), allow_http=True).read_bytes() == data


def test_download_from_base_installs_and_keeps_previous(tmp_path):
    pack1, pack2 = tmp_path / "srv" / "v1", tmp_path / "srv2" / "v1"
    make_pack(pack1)
    make_pack(pack2, payload=b"new-")
    models, dl = tmp_path / "data" / "models", tmp_path / "data" / "downloads"
    seen = []
    with Server(tmp_path / "srv") as s:
        names = modelpack.download_from_base(s.url + "v1/", models, dl, lambda *a: seen.append(a), allow_http=True)
    assert "eye.onnx.data" in names and (models / "manifest.json").is_file()
    assert (models / "detector.onnx").read_bytes() == b"model-bytes" + b"detector.onnx"
    assert any(w == "downloading" for w, *_ in seen) and seen[-1][0] in ("downloading", "verifying")
    assert not modelpack.previous(models).exists()
    with Server(tmp_path / "srv2") as s:
        modelpack.download_from_base(s.url + "v1/", models, dl, allow_http=True)
    assert (models / "detector.onnx").read_bytes().startswith(b"new-")
    assert (modelpack.previous(models) / "detector.onnx").read_bytes().startswith(b"model-")  # kept for rollback
    modelpack.rollback(models)
    assert (models / "detector.onnx").read_bytes().startswith(b"model-")
    assert not list(models.parent.glob("models.new-*")) and not list(dl.glob("*.part"))


def test_damaged_download_leaves_old_models_alone(tmp_path):
    good = tmp_path / "srv" / "v1"
    files = make_pack(good)
    models, dl = tmp_path / "data" / "models", tmp_path / "data" / "downloads"
    with Server(tmp_path / "srv") as s:
        modelpack.download_from_base(s.url + "v1/", models, dl, allow_http=True)
    before = (models / "eye.onnx").read_bytes()
    bad = tmp_path / "srv2" / "v1"
    make_pack(bad, payload=b"zzzz")
    manifest = json.loads((bad / "manifest.json").read_text())
    manifest["files"]["eye.onnx"]["sha256"] = sha(b"not what is served")  # the manifest lies about one file
    (bad / "manifest.json").write_text(json.dumps(manifest))
    with Server(tmp_path / "srv2") as s, pytest.raises(modelpack.PackError, match=r"eye\.onnx"):
        modelpack.download_from_base(s.url + "v1/", models, dl, allow_http=True)
    assert (models / "eye.onnx").read_bytes() == before and files
    assert not list(models.parent.glob("models.new-*"))


@pytest.mark.parametrize("status,kind", [(401, "private"), (403, "private"), (404, "missing"), (500, "other")])
def test_source_errors_are_told_apart(tmp_path, status, kind):
    with Server(tmp_path, status=status) as s, pytest.raises(modelpack.SourceError) as e:
        modelpack.download_from_base(s.url + "v1/", tmp_path / "m", tmp_path / "d", allow_http=True)
    assert e.value.kind == kind and str(status) in str(e.value)


def test_no_connection_is_a_network_error(tmp_path):
    with Server(tmp_path) as s:
        url = s.url
    with pytest.raises(modelpack.SourceError) as e:
        modelpack.download_from_base(url + "v1/", tmp_path / "m", tmp_path / "d", allow_http=True)
    assert e.value.kind == "network"


def test_manifest_with_bad_names_is_refused(tmp_path):
    srv = tmp_path / "srv" / "v1"
    srv.mkdir(parents=True)
    (srv / "manifest.json").write_text(json.dumps({"files": {"../evil.onnx": {"bytes": 3, "sha256": sha(b"abc")}}}))
    with Server(tmp_path / "srv") as s, pytest.raises(modelpack.PackError, match="not a model pack file"):
        modelpack.download_from_base(s.url + "v1/", tmp_path / "m", tmp_path / "d", allow_http=True)
    assert not (tmp_path / "evil.onnx").exists()


@pytest.mark.parametrize(
    "name", ["../x.onnx", "heads/fast_h6.txt", "a/b.onnx", "x.exe", "x.dll", "", "x.onnx/", "-x.onnx", "x" * 100 + ".onnx"]
)
def test_check_name_refuses(name):
    with pytest.raises(modelpack.PackError):
        modelpack.check_name(name)


def test_zip_and_folder_installs_skip_heads(tmp_path):
    import zipfile

    src = tmp_path / "pack"
    make_pack(src)
    (src / "heads").mkdir()
    (src / "heads" / "fast_h6.txt").write_text("1;0;1")  # an older pack that carried the heads: skipped
    z = tmp_path / "pack.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, "models/" + p.relative_to(src).as_posix())  # zipped as one folder: accepted
    models = tmp_path / "data" / "models"
    modelpack.install_zip(z, models)
    assert (models / "eye.onnx").is_file() and not (models / "heads").exists()
    models2 = tmp_path / "data2" / "models"
    modelpack.install_folder(src, models2)
    assert (models2 / "eye.onnx").is_file() and not (models2 / "heads").exists()


def test_bad_zip_and_zip_slip_are_refused(tmp_path):
    import zipfile

    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(modelpack.PackError):
        modelpack.install_zip(bad, tmp_path / "m")
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("../evil.onnx", "x")
    with pytest.raises(modelpack.PackError):
        modelpack.install_zip(evil, tmp_path / "m")
    assert not (tmp_path / "evil.onnx").exists() and not (tmp_path / "m").exists()


def test_real_verify_refuses_a_pack_without_the_required_models(tmp_path):
    make_pack(tmp_path / "p", names=("eye.onnx",))
    with pytest.raises(modelpack.PackError, match="lacks"):
        REAL_VERIFY(tmp_path / "p")


def test_problem_reports_missing_models(tmp_path):
    assert "not installed" in modelpack.problem(tmp_path / "none")
