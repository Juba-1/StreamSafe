"""Temporary Pro pass for supporters (launch period, until the paid plans go live).

Ali gives a supporter a key that turns Pro on for a fixed time (30 days by default), then it stops by itself.
The key is a small signed note, checked here with the public key only (``release.PASS_PUBLIC_KEYS``), so it works
without a server or the internet:

    SSP-<base64url of the JSON note>.<base64url of the ed25519 signature of that note>

The note: {"p": "pro", "e": "YYYY-MM-DD" (last day it works), "w": "who it was made for", "i": "short id"}.
Keys are made by ``tools/make_pass.py`` (the GitHub workflow make-pass), never in the app.
"""

from __future__ import annotations

import base64
import contextlib
import datetime as dt
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from streamsafe.app import ed25519, paths, release

PREFIX = "SSP-"


@dataclass(frozen=True)
class Pass:
    plan: str
    expires: dt.date  # the last day it works
    who: str
    pid: str
    key: str


def _b64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _today() -> dt.date:
    return dt.datetime.now().date()


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
        return Pass(str(d["p"]), dt.date.fromisoformat(str(d["e"])), str(d.get("w", "")), str(d.get("i", "")), key)
    except (ValueError, KeyError, TypeError):
        return None


def _file(env: Mapping[str, str] | None = None):
    return paths.data_dir(dict(env) if env is not None else None) / "pass.json"


def load(env: Mapping[str, str] | None = None) -> Pass | None:
    """The saved pass, checked again (signature) every time; None if there is none or it is broken."""
    try:
        return parse(json.loads(_file(env).read_text(encoding="utf-8"))["key"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def active(env: Mapping[str, str] | None = None, today: Callable[[], dt.date] = _today) -> Pass | None:
    p = load(env)
    return p if p is not None and today() <= p.expires else None


def activate(key: str, env: Mapping[str, str] | None = None, today: Callable[[], dt.date] = _today) -> str:
    """Saves a good pass. Returns "" or the reason: invalid / expired."""
    p = parse(key)
    if p is None:
        return "invalid"
    if today() > p.expires:
        return "expired"
    f = _file(env)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"key": p.key}), encoding="utf-8")
    return ""


def forget(env: Mapping[str, str] | None = None) -> None:
    with contextlib.suppress(OSError):
        _file(env).unlink()
