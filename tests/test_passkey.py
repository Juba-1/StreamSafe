"""Supporter Pro pass: one month, then off for good; the clock cannot bring it back."""

import base64
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from streamsafe.app import passkey, plan, release

D = dt.date


def b64(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


@pytest.fixture
def k(monkeypatch):
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(release, "PASS_PUBLIC_KEYS", (key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),))
    monkeypatch.setattr(passkey, "internet_today", lambda *a, **kw: None)  # no network in tests
    return key


def make(key, last="2026-12-17", who="ali"):
    note = json.dumps({"p": "pro", "e": last, "w": who, "i": "abc"}).encode()
    return f"SSP-{b64(note)}.{b64(key.sign(note))}"


@pytest.fixture
def env(tmp_path):
    return {"STREAMSAFE_DATA_DIR": str(tmp_path), "LOCALAPPDATA": str(tmp_path)}


def day(y, m, d):
    return lambda: D(y, m, d)


def test_month_then_off_by_itself(k, env):
    assert plan.current(env) == "basic"
    assert passkey.activate(make(k), env, today=day(2026, 11, 17)) == ""
    assert passkey.why_off(env, today=day(2026, 12, 17)) == ""  # the last day still works
    assert passkey.why_off(env, today=day(2026, 12, 18)) == "expired"  # then it stops by itself


def test_pro_follows_the_pass(k, env, monkeypatch):
    monkeypatch.setattr(passkey, "_local_today", lambda: D(2026, 11, 20))
    passkey.activate(make(k), env)
    assert plan.current(env) == "pro" and plan.allows("custom_brand", env)
    monkeypatch.setattr(passkey, "_local_today", lambda: D(2027, 1, 1))
    assert plan.current(env) == "basic"


def test_cannot_be_used_again_after_it_ends(k, env):
    key = make(k)
    passkey.activate(key, env, today=day(2026, 11, 17))
    passkey.why_off(env, today=day(2026, 12, 20))  # the app ran after the end
    passkey.forget(env)  # "deactivate" and paste it again
    assert passkey.activate(key, env, today=day(2026, 12, 20)) == "expired"
    assert passkey.activate(key, env, today=day(2026, 12, 1)) == "clock"  # ... with the clock set back
    assert passkey.why_off(env, today=day(2026, 12, 1)) == "none"


def test_clock_set_back_turns_pro_off(k, env):
    passkey.activate(make(k), env, today=day(2026, 11, 17))
    passkey.why_off(env, today=day(2026, 12, 10))
    assert passkey.why_off(env, today=day(2026, 11, 20)) == "clock"
    assert passkey.why_off(env, today=day(2026, 12, 9)) == ""  # one day of slack (time zones)


def test_internet_date_counts(k, env):
    passkey.activate(make(k), env, today=day(2026, 11, 17))
    passkey.sync_clock(env, fetch=lambda: D(2026, 12, 25))  # the PC clock says November, the internet says after the end
    assert passkey.why_off(env, today=day(2026, 11, 18)) == "clock"
    assert passkey.activate(make(k), env, today=day(2026, 11, 18), net=lambda: D(2026, 12, 25)) == "clock"
    fresh = {"STREAMSAFE_DATA_DIR": env["STREAMSAFE_DATA_DIR"] + "_new"}
    assert passkey.activate(make(k), fresh, today=day(2026, 11, 18), net=lambda: D(2026, 12, 25)) == "expired"


def test_changed_key_or_other_signer_refused(k, env):
    key = make(k)
    body, sig = key[4:].split(".")
    note = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    note["e"] = "2099-01-01"
    assert passkey.activate("SSP-" + b64(json.dumps(note).encode()) + "." + sig, env) == "invalid"
    assert passkey.activate(make(Ed25519PrivateKey.generate()), env) == "invalid"
    assert passkey.activate("SSP-garbage", env) == "invalid"
    assert passkey.is_pass(" ssp-x ") and not passkey.is_pass("ABCD-1234")


def test_pasted_with_spaces_and_line_breaks(k, env):
    key = make(k, last="2030-01-01")
    assert passkey.activate(key[:20] + "\n " + key[20:40] + " " + key[40:], env, today=day(2026, 11, 17)) == ""


def test_make_pass_tool_round_trip(k, env, monkeypatch, tmp_path):
    """The real tool makes a key the app accepts."""
    tool = Path(__file__).resolve().parents[1] / "tools" / "make_pass.py"
    priv = k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()
    out = subprocess.run(
        [sys.executable, str(tool), "--who", "Sara", "--days", "30"],
        env={**os.environ, "PASS_SIGNING_KEY": priv},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert passkey.activate(out[0], env) == ""
    p = passkey.active(env)
    assert p is not None and p.who == "Sara" and p.expires == dt.date.today() + dt.timedelta(days=30)


def test_streamer_logo_goes_back_to_ours_when_the_pass_ends(k, env, monkeypatch, tmp_path):
    from streamsafe.app import engine, settings

    logo = tmp_path / "mine.png"
    logo.write_bytes(b"png")
    s = settings.Settings(brand_image=str(logo))
    monkeypatch.setattr(passkey, "_local_today", lambda: D(2026, 11, 20))
    passkey.activate(make(k), env)
    on = engine.engine_env(dict(env), False, "", env, brand_image=s.brand_image)
    assert on["LOGO"] == str(logo) and on["SCENE_IMAGE"] == str(logo)
    monkeypatch.setattr(passkey, "_local_today", lambda: D(2027, 1, 1))  # the month is over: next start
    off = engine.engine_env(dict(env), False, "", env, brand_image=s.brand_image)
    assert off.get("LOGO") != str(logo) and off.get("SCENE_IMAGE") != str(logo)
