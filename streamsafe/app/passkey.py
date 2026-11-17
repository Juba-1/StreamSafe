"""Temporary Pro pass for supporters (launch period, until the paid plans go live).

Ali gives a supporter a key that turns Pro on for a fixed time (30 days by default), then it stops by itself.
The key is a small signed note, checked here with the public key only (``release.PASS_PUBLIC_KEYS``), so it works
without a server:

    SSP-<base64url of the JSON note>.<base64url of the ed25519 signature of that note>

The note: {"p": "pro", "e": "YYYY-MM-DD" (last day it works), "d": "device code", "w": "who", "i": "short id"}.

Rules the key cannot get around without breaking the signature:
- one PC: the note carries the device code the supporter's app shows (from Windows' machine id); on another PC it
  is refused, and a copied ``pass.json`` does not work there either;
- one time: after its last day it is refused for good (also when pasted again);
- the clock: the latest date the app has seen (its own clock, and the internet's clock whenever it can ask) is
  kept; a PC clock set back before that date turns the pass off instead of making it last longer.

Keys are made by ``tools/make_pass.py`` (the make-pass workflow), never in the app.
"""

from __future__ import annotations

import base64
import contextlib
import datetime as dt
import email.utils
import hashlib
import json
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from streamsafe.app import ed25519, paths, release

PREFIX = "SSP-"
TIME_URL = "https://www.google.com/generate_204"  # any big site's Date header is the internet's clock
Today = Callable[[], dt.date]


@dataclass(frozen=True)
class Pass:
    plan: str
    expires: dt.date  # the last day it works
    device: str  # the device code it was made for
    who: str
    pid: str
    key: str


# --- this PC -------------------------------------------------------------------------------------------------
def machine_id() -> str:
    """Windows' own id of this installation (stays the same across app updates and reinstalls)."""
    with contextlib.suppress(Exception):
        import winreg  # type: ignore[import-not-found]

        flags = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0, flags) as k:
            return str(winreg.QueryValueEx(k, "MachineGuid")[0])
    return f"node-{uuid.getnode():012x}"


def clean_code(code: str) -> str:
    return "".join(ch for ch in code.upper() if ch.isalnum())


def device_code(mid: str | None = None) -> str:
    """What the supporter sends with their support: 8 letters/digits, shown as XXXX-XXXX."""
    h = hashlib.sha256(("streamsafe-device:" + (mid if mid is not None else machine_id())).encode()).digest()
    s = base64.b32encode(h).decode()[:8]
    return f"{s[:4]}-{s[4:]}"


# --- the key -------------------------------------------------------------------------------------------------
def _b64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def is_pass(key: str) -> bool:
    return "".join(key.split()).upper().startswith(PREFIX)


def parse(key: str) -> Pass | None:
    """The pass if the signature is ours, else None (any change to the note breaks the signature)."""
    key = "".join(key.split())
    if not key.upper().startswith(PREFIX) or "." not in key:
        return None
    body, _, sig = key[len(PREFIX) :].partition(".")
    try:
        note, signature = _b64(body), _b64(sig)
        if not any(ed25519.verify(bytes.fromhex(k), note, signature) for k in release.PASS_PUBLIC_KEYS):
            return None
        d = json.loads(note.decode("utf-8"))
        return Pass(str(d["p"]), dt.date.fromisoformat(str(d["e"])), clean_code(str(d["d"])), str(d.get("w", "")), str(d.get("i", "")), key)
    except (ValueError, KeyError, TypeError):
        return None


# --- what this PC holds: the key and the latest date seen ------------------------------------------------------
def _file(env: Mapping[str, str] | None = None):
    return paths.data_dir(dict(env) if env is not None else None) / "pass.json"


def _read(env: Mapping[str, str] | None) -> dict:
    try:
        d = json.loads(_file(env).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(env: Mapping[str, str] | None, d: dict) -> None:
    f = _file(env)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d), encoding="utf-8")


def _seen(d: dict) -> dt.date | None:
    with contextlib.suppress(KeyError, TypeError, ValueError):
        return dt.date.fromisoformat(str(d["seen"]))
    return None


def _local_today() -> dt.date:
    return dt.datetime.now().date()


def load(env: Mapping[str, str] | None = None) -> Pass | None:
    """The saved pass, its signature checked again every time; None if there is none or it is broken."""
    return parse(str(_read(env).get("key", "")))


def why_off(env: Mapping[str, str] | None = None, today: Today | None = None, mid: str | None = None) -> str:
    """ "" while the saved pass gives Pro, else why not: none / other_device / clock / expired.
    Also moves the latest date seen forward (never back)."""
    d = _read(env)
    p = parse(str(d.get("key", "")))
    if p is None:
        return "none"
    if p.device != clean_code(device_code(mid)):
        return "other_device"
    now, seen = (today or _local_today)(), _seen(d)
    if seen is not None and now < seen - dt.timedelta(days=1):  # the PC clock was set back (one day of slack: time zones)
        return "clock"
    if seen is None or now > seen:
        d["seen"] = now.isoformat()
        with contextlib.suppress(OSError):
            _write(env, d)
    return "expired" if max(now, seen or now) > p.expires else ""


def active(env: Mapping[str, str] | None = None, today: Today | None = None, mid: str | None = None) -> Pass | None:
    return load(env) if why_off(env, today, mid) == "" else None


def activate(
    key: str,
    env: Mapping[str, str] | None = None,
    today: Today | None = None,
    mid: str | None = None,
    net: Callable[[], dt.date | None] | None = None,
) -> str:
    """Saves a good pass for this PC. Returns "" or the reason: invalid / other_device / expired / clock.
    Network (call it off the window's thread): the internet's date counts too, when it can be had."""
    p = parse(key)
    if p is None:
        return "invalid"
    if p.device != clean_code(device_code(mid)):
        return "other_device"
    d = _read(env)
    now, seen = (today or _local_today)(), _seen(d)
    if seen is not None and now < seen - dt.timedelta(days=1):
        return "clock"
    fetch = net or internet_today
    latest = max(x for x in (now, seen, fetch()) if x is not None)
    if latest > p.expires:
        return "expired"
    _write(env, {"key": p.key, "seen": latest.isoformat()})
    return ""


def forget(env: Mapping[str, str] | None = None) -> None:
    """Removes the key but keeps the latest date seen (so an ended key cannot come back with the clock)."""
    d = _read(env)
    if not d:
        return
    d.pop("key", None)
    with contextlib.suppress(OSError):
        _write(env, d)


# --- the internet's clock --------------------------------------------------------------------------------------
def internet_today(url: str = TIME_URL, timeout: float = 5.0) -> dt.date | None:
    """Today's date from a big site's Date header (UTC); None without internet."""
    with contextlib.suppress(Exception):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return email.utils.parsedate_to_datetime(r.headers["Date"]).date()
    return None


def sync_clock(env: Mapping[str, str] | None = None, fetch: Callable[[], dt.date | None] | None = None) -> None:
    """Network (call it off the window's thread): the latest date seen follows the internet's clock, so a PC
    clock set back cannot stretch a pass. Only for a PC that holds a pass."""
    d = _read(env)
    if "key" not in d:
        return
    net = (fetch or internet_today)()
    seen = _seen(d)
    if net is not None and (seen is None or net > seen):
        d["seen"] = net.isoformat()
        with contextlib.suppress(OSError):
            _write(env, d)
