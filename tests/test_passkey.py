import base64
import datetime as dt
import json

import pytest

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat  # noqa: E402

from streamsafe.app import passkey, plan, release  # noqa: E402


def b64(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


@pytest.fixture
def signer(monkeypatch):
    k = Ed25519PrivateKey.generate()
    monkeypatch.setattr(release, "PASS_PUBLIC_KEYS", (k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),))

    def make(last, plan_="pro", who="ali"):
        note = json.dumps({"p": plan_, "e": last, "w": who, "i": "abc"}).encode()
        return f"SSP-{b64(note)}.{b64(k.sign(note))}"

    return make


def env(tmp_path):
    return {"STREAMSAFE_HOME": str(tmp_path), "LOCALAPPDATA": str(tmp_path), "APPDATA": str(tmp_path)}


def test_pass_turns_pro_on_then_off(signer, tmp_path):
    e = env(tmp_path)
    key = signer("2026-12-17")
    assert plan.current(e) == "basic"
    assert passkey.activate(key, e, today=lambda: dt.date(2026, 11, 17)) == ""
    assert passkey.active(e, today=lambda: dt.date(2026, 12, 17)) is not None  # the last day still works
    assert passkey.active(e, today=lambda: dt.date(2026, 12, 18)) is None  # then it stops by itself
    passkey.forget(e)
    assert passkey.load(e) is None


def test_changed_or_foreign_key_is_refused(signer, tmp_path):
    e = env(tmp_path)
    key = signer("2026-12-17")
    body, sig = key[4:].split(".")
    note = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    note["e"] = "2099-01-01"  # someone edits the date
    forged = "SSP-" + b64(json.dumps(note).encode()) + "." + sig
    assert passkey.activate(forged, e) == "invalid"
    assert passkey.activate("SSP-garbage", e) == "invalid"
    assert passkey.activate(signer("2020-01-01"), e) == "expired"
    assert passkey.is_pass(" ssp-x ") and not passkey.is_pass("ABCD-1234")


def test_real_key_shape_with_spaces(signer, tmp_path):
    e = env(tmp_path)
    key = signer("2030-01-01")
    assert passkey.activate(key[:20] + "\n " + key[20:], e) == ""  # pasted with a line break
