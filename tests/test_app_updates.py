"""Signature (ed25519), beta expiry, the signed update manifest, and data-only updates with rollback."""

import datetime as dt
import json

import pytest

from streamsafe.app import build_info, ed25519, heads, modelpack, release, updates
from tests.appfix import Server, light_verify, make_heads, make_pack, sha

# RFC 8032 section 7.1 test vectors (public key, message, signature)
RFC8032 = [
    (
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "fc51cd8e6218a0a3b7b7f0e6e0d7f2f5e5c6a8d0a0e0e0e0e0e0e0e0e0e0e0e0",  # not a valid key: must be refused, not crash
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
]


def test_ed25519_rfc8032_vectors_and_tampering():
    pub, msg, sig = (bytes.fromhex(x) for x in RFC8032[0])
    assert ed25519.verify(pub, msg, sig)
    assert not ed25519.verify(pub, b"x", sig)
    assert not ed25519.verify(pub, msg, sig[:-1] + bytes([sig[-1] ^ 1]))
    pub2, msg2, sig2 = (bytes.fromhex(x) for x in RFC8032[1])
    assert ed25519.verify(pub2, msg2, sig2) and not ed25519.verify(pub, msg2, sig2)
    bad_pub, msg3, sig3 = (bytes.fromhex(x) for x in RFC8032[2])
    assert ed25519.verify(bad_pub, msg3, sig3) is False
    assert not ed25519.verify(b"short", msg, sig) and not ed25519.verify(pub, msg, b"short")


# --- expiry ---------------------------------------------------------------------------------------------------
def test_expiry_dates():
    day = dt.date(2026, 12, 1)
    assert updates.days_left(day, "") is None and not updates.expired(day, "")  # a source copy never expires
    assert updates.days_left(day, "2026-12-31") == 30 and not updates.expired(day, "2026-12-31")
    assert updates.expired(day, "2026-12-01")  # the day itself is the first day it no longer starts
    assert updates.expired(day, "2026-11-30") and updates.days_left(day, "2026-11-30") == -1
    assert updates.expired(day, "garbage") and updates.days_left(day, "garbage") == 0  # a damaged stamp is expired


def test_build_stamp_of_a_source_copy_never_expires():
    assert build_info.EXPIRES == "" or updates.days_left() is not None


def test_beta_length_default():
    assert release.BETA_DAYS == 0  # paid builds never end (17 Nov 2026)


# --- signed manifest ------------------------------------------------------------------------------------------
@pytest.fixture
def signer(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()

    def sign(payload: dict, k=key) -> bytes:
        text = json.dumps(payload)
        return json.dumps({"payload": text, "signatures": [k.sign(text.encode()).hex()]}).encode()

    return pub, sign, Ed25519PrivateKey


def payload(serial=5, **extra):
    return {"format": 1, "product": "streamsafe-app", "serial": serial, **extra}


def test_signed_manifest_is_accepted(signer):
    pub, sign, _ = signer
    u = updates.parse_signed(sign(payload(app={"version": "3.1.0", "url": "https://x.example/a", "notes_en": "hi"})), (pub,))
    assert u.serial == 5 and u.app_version == "3.1.0" and u.notes["en"] == "hi"
    assert u.app_is_newer("3.0.0.9") and not u.app_is_newer("3.1.0") and not u.app_is_newer("3.2")


def test_wrong_key_tampered_and_unsigned_are_refused(signer):
    pub, sign, Key = signer
    other = Key.generate()
    with pytest.raises(updates.UpdateError, match="not signed by us"):
        updates.parse_signed(sign(payload(), other), (pub,))
    env = json.loads(sign(payload(app={"version": "3.1.0"})))
    env["payload"] = env["payload"].replace("3.1.0", "9.9.9")
    with pytest.raises(updates.UpdateError, match="not signed by us"):
        updates.parse_signed(json.dumps(env).encode(), (pub,))
    with pytest.raises(updates.UpdateError, match="not readable"):
        updates.parse_signed(b"{}", (pub,))
    with pytest.raises(updates.UpdateError, match="not readable"):
        updates.parse_signed(b"\xff\xfe", (pub,))
    with pytest.raises(updates.UpdateError, match="no update key"):
        updates.parse_signed(sign(payload()), ())


def test_replay_of_an_older_manifest_and_foreign_product_are_refused(signer):
    pub, sign, _ = signer
    with pytest.raises(updates.UpdateError, match="older"):
        updates.parse_signed(sign(payload(serial=4)), (pub,), min_serial=5)
    updates.parse_signed(sign(payload(serial=5)), (pub,), min_serial=5)
    with pytest.raises(updates.UpdateError, match="not usable"):
        updates.parse_signed(sign({**payload(), "product": "other"}), (pub,))
    with pytest.raises(updates.UpdateError, match="not usable"):
        updates.parse_signed(sign({**payload(), "format": 2}), (pub,))


def test_manifest_with_http_or_odd_urls_is_refused(signer):
    pub, sign, _ = signer
    for url in ("http://x.example/a", "file:///etc/passwd", "ftp://x/a"):
        with pytest.raises(updates.UpdateError, match="not usable"):
            updates.parse_signed(sign(payload(app={"version": "3.1.0", "url": url})), (pub,))
    f = {"x.txt": {"url": "http://x.example/x", "sha256": "0" * 64, "bytes": 1}}
    with pytest.raises(updates.UpdateError, match="not usable"):
        updates.parse_signed(sign(payload(heads={"version": 2, "files": f})), (pub,))


def test_update_is_off_without_url_or_key_and_never_raises_into_the_app(monkeypatch, signer):
    pub, sign, _ = signer
    monkeypatch.setattr(release, "UPDATE_MANIFEST_URL", "")
    monkeypatch.setattr(release, "UPDATE_PUBLIC_KEYS", ())
    assert updates.check() is None  # the app works normally
    monkeypatch.setattr(release, "UPDATE_MANIFEST_URL", "https://updates.example/manifest.json")
    assert updates.check() is None  # a URL but no key
    monkeypatch.setattr(release, "UPDATE_PUBLIC_KEYS", (pub,))
    from streamsafe.app import fetch

    def unreachable(url):
        raise fetch.DownloadError("download failed: timed out")

    with pytest.raises(updates.UpdateError):  # reported, the caller carries on
        updates.check(fetcher=unreachable)
    assert updates.check(fetcher=lambda url: sign(payload(app={"version": "9.0.0"}))).app_version == "9.0.0"


def test_sign_tool_roundtrip(tmp_path, signer):
    from tools import sign_manifest

    key = tmp_path / "private.pem"
    assert sign_manifest.main(["keygen", "--private", str(key)]) == 0
    assert sign_manifest.main(["keygen", "--private", str(key)]) == 2  # never overwrites a key
    models = tmp_path / "m"
    make_pack(models)
    hd = make_heads(tmp_path / "h", version=3)
    out = tmp_path / "manifest.json"
    args = [
        "sign",
        "--private",
        str(key),
        "--serial",
        "7",
        "--out",
        str(out),
        "--app-version",
        "3.2.0",
        "--app-url",
        "https://x.example/get",
    ]
    args += ["--models-dir", str(models), "--models-url", "https://x.example/models/", "--models-version", "v2"]
    args += ["--heads-dir", str(hd), "--heads-url", "https://x.example/heads/", "--heads-version", "3"]
    assert sign_manifest.main(args) == 0
    from cryptography.hazmat.primitives import serialization

    public = serialization.load_pem_private_key(key.read_bytes(), None).public_key()
    pub_hex = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    u = updates.parse_signed(out.read_bytes(), (pub_hex,))
    assert u.serial == 7 and u.heads_version == 3 and set(u.head_files) == {"fast_h6.txt", "male_h6.txt", "thr_h6.txt"}
    assert u.model_files["manifest.json"]["sha256"] == sha((models / "manifest.json").read_bytes())
    assert u.model_files["eye.onnx"]["url"] == "https://x.example/models/eye.onnx"
    assert not updates.parse_signed(out.read_bytes(), (pub_hex,), 7).app_is_newer("3.2.0")


# --- applying data updates ----------------------------------------------------------------------------------
def served(tmp_path, models_payload=b"new-", heads_version=2):
    srv = tmp_path / "srv"
    mp = srv / "models"
    make_pack(mp, payload=models_payload)
    hp = make_heads(srv / "heads", version=heads_version, n=12)
    return srv, mp, hp


def update_for(base, mp, hp, heads_version=2):
    def entry(folder, name, sub):
        data = (folder / name).read_bytes()
        return {"url": f"{base}{sub}/{name}", "sha256": sha(data), "bytes": len(data)}

    mfiles = ["manifest.json", *json.loads((mp / "manifest.json").read_text())["files"]]
    return updates.Update(
        serial=9,
        models_version="v2",
        model_files={n: entry(mp, n, "models") for n in mfiles},
        heads_version=heads_version,
        head_files={n: entry(hp, n, "heads") for n in ("fast_h6.txt", "male_h6.txt", "thr_h6.txt")},
    )


def test_data_update_swaps_atomically_keeps_previous_and_rolls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(modelpack, "verify", light_verify)
    models, hfolder, dl = tmp_path / "data" / "models", tmp_path / "data" / "heads", tmp_path / "data" / "dl"
    make_pack(models, payload=b"old-")
    srv, mp, hp = served(tmp_path)
    with Server(srv) as s:
        u = update_for(s.url, mp, hp)
        assert updates.models_differ(u, modelpack.current_shas(models)) and updates.heads_newer(u, 1)
        done = updates.apply_data(u, models=models, heads_folder=hfolder, downloads=dl, installed_heads_version=1, allow_http=True)
    assert done == ["models", "heads"]
    assert (models / "eye.onnx").read_bytes().startswith(b"new-") and (modelpack.previous(models) / "eye.onnx").read_bytes().startswith(
        b"old-"
    )
    assert heads.version(hfolder) == 2 and not heads.problem(hfolder)
    assert not updates.models_differ(u, modelpack.current_shas(models))  # nothing left to do
    assert not updates.heads_newer(u, 2)  # only a HIGHER heads version replaces


def test_heads_update_keeps_the_previous_set_and_rolls_back(tmp_path):
    hfolder, dl = tmp_path / "data" / "heads", tmp_path / "dl"
    make_heads(hfolder, version=2, n=8)
    srv, _, hp = served(tmp_path, heads_version=3)
    with Server(srv) as s:
        wanted = {
            n: {"url": f"{s.url}heads/{n}", "sha256": sha((hp / n).read_bytes()), "bytes": len((hp / n).read_bytes())} for n in heads.FILES
        }
        assert heads.install_files(wanted, 3, hfolder, dl, allow_http=True) == sorted(heads.FILES)
    assert heads.version(hfolder) == 3 and heads.version(heads.previous(hfolder)) == 2
    heads.rollback(hfolder)
    assert heads.version(hfolder) == 2 and heads.version(heads.previous(hfolder)) == 3


def test_bad_sha_or_unloadable_heads_change_nothing(tmp_path):
    hfolder, dl = tmp_path / "data" / "heads", tmp_path / "dl"
    make_heads(hfolder, version=2, n=8)
    srv, _, hp = served(tmp_path, heads_version=3)
    (srv / "heads" / "thr_h6.txt").write_text("7.5")  # served file differs from what the manifest pinned
    with Server(srv) as s:
        wanted = {
            n: {"url": f"{s.url}heads/{n}", "sha256": sha((hp / n).read_bytes()), "bytes": len((hp / n).read_bytes())} for n in heads.FILES
        }
        with pytest.raises(heads.HeadsError):
            heads.install_files(wanted, 3, hfolder, dl, allow_http=True)
        # a correctly hashed set that does not load (threshold out of range) is refused too
        wanted = {
            n: {"url": f"{s.url}heads/{n}", "sha256": sha((srv / "heads" / n).read_bytes()), "bytes": len((srv / "heads" / n).read_bytes())}
            for n in heads.FILES
        }
        with pytest.raises(heads.HeadsError, match="do not load"):
            heads.install_files(wanted, 3, hfolder, dl, allow_http=True)
    assert heads.version(hfolder) == 2 and not heads.problem(hfolder)
    assert not list(hfolder.parent.glob("heads.new-*"))


def test_heads_install_refuses_unexpected_file_sets(tmp_path):
    with pytest.raises(heads.HeadsError):
        heads.install_files({"fast_h6.txt": {}}, 2, tmp_path / "h", tmp_path / "d")
    with pytest.raises(heads.HeadsError):
        heads.install_files({n: {} for n in (*heads.FILES, "evil.dll")}, 2, tmp_path / "h", tmp_path / "d")


def test_heads_best_prefers_higher_version_then_installed_and_refuses_placeholder(tmp_path):
    shipped, env = tmp_path / "app" / "heads", {"STREAMSAFE_DATA_DIR": str(tmp_path / "data")}
    updated = tmp_path / "data" / "heads"
    assert heads.best(env, shipped) is None and heads.why_not(env, shipped) == "missing"
    make_heads(shipped, version=2)
    assert heads.best(env, shipped) == shipped
    make_heads(updated, version=2)
    assert heads.best(env, shipped) == shipped  # tie: the installed ones
    make_heads(updated, version=3)
    assert heads.best(env, shipped) == updated
    make_heads(shipped, version=4)  # a newer installer: its heads win over an older update
    assert heads.best(env, shipped) == shipped
    (updated / "thr_h6.txt").write_text("nonsense")  # damaged update: never used
    make_heads(shipped, version=1)
    assert heads.best(env, shipped) == shipped
    (shipped / heads.PLACEHOLDER).write_text("test")
    assert heads.best(env, shipped) is None and heads.why_not(env, shipped) == "placeholder"
