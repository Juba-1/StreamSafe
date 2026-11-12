"""Beta expiry and the update check.

Expiry: a beta build carries the date after which it no longer STARTS a stream (``build_info.EXPIRES``, set by
the Windows build). Refusing to start is the safe side: an expired build never streams at all, so it can never
stream uncovered. A stream that is already running is not cut.

Updates: when ``release.UPDATE_MANIFEST_URL`` is set, the app fetches one small JSON file on start:

    {"payload": "<JSON text>", "signatures": ["<hex ed25519 signature of the payload text's UTF-8 bytes>", ...]}

    payload = {"format": 1, "product": "streamsafe-app", "serial": 12,
               "app":    {"version": "3.0.1", "url": "https://...", "notes_ar": "...", "notes_en": "..."},
               "models": {"version": "2026-11-20", "files": {"manifest.json": {"url": ..., "sha256": ..., "bytes": ...}, ...}},
               "heads":  {"version": 3, "files": {"fast_h6.txt": {"url": ..., "sha256": ..., "bytes": ...}, ...}}}

It is used only if a signature verifies under a public key built into the app (``release.UPDATE_PUBLIC_KEYS``),
and only if its serial is not lower than the last one seen (an old, validly signed file cannot be replayed to
bring old models back). The manifest carries DATA only: a new app version is shown to the streamer as a link
they open themselves; model files are downloaded, verified and swapped by ``modelpack.install_files``.
Nothing downloaded is ever executed. Any failure leaves everything as it is.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from streamsafe import __version__
from streamsafe.app import build_info, ed25519, fetch, heads, modelpack, release

FORMAT = 1
PRODUCT = "streamsafe-app"
MAX_MANIFEST_BYTES = 256 << 10


class UpdateError(Exception):
    """The update manifest could not be fetched or must not be trusted. Nothing was changed."""


# --- expiry -------------------------------------------------------------------------------------------------
def days_left(today: dt.date | None = None, expires: str | None = None) -> int | None:
    """Days this build may still start a stream (0 or less: expired); None = this build never expires.
    A date that cannot be read counts as expired: a damaged build stamp must not make a beta immortal."""
    expires = build_info.EXPIRES if expires is None else expires
    if not expires:
        return None
    try:
        end = dt.date.fromisoformat(expires)
    except ValueError:
        return 0
    return (end - (today or dt.date.today())).days


def expired(today: dt.date | None = None, expires: str | None = None) -> bool:
    left = days_left(today, expires)
    return left is not None and left <= 0


# --- update manifest ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Update:
    serial: int
    app_version: str = ""  # "" = the manifest names no app version
    app_url: str = ""
    notes: dict[str, str] = field(default_factory=dict)  # {"ar": ..., "en": ...}
    models_version: str = ""
    model_files: dict[str, dict] = field(default_factory=dict)  # name -> {"url", "sha256", "bytes"}
    heads_version: int = 0
    head_files: dict[str, dict] = field(default_factory=dict)

    def app_is_newer(self, current: str = "") -> bool:
        return bool(self.app_version) and version_key(self.app_version) > version_key(current or current_version())


def current_version() -> str:
    """The running build's version with its build number (``3.0.0.57``); from source ``streamsafe.__version__``."""
    return build_info.VERSION or __version__


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", version)[:4])


def parse_signed(raw: bytes, keys: tuple[str, ...], min_serial: int = 0) -> Update:
    """The verified content of a signed manifest, or UpdateError."""
    try:
        envelope = json.loads(raw.decode("utf-8"))
        payload_text, signatures = envelope["payload"], envelope["signatures"]
        if not isinstance(payload_text, str) or not isinstance(signatures, list):
            raise TypeError("wrong types")
        message = payload_text.encode("utf-8")
        public = [bytes.fromhex(k) for k in keys]
        sigs = [bytes.fromhex(s) for s in signatures[:8] if isinstance(s, str)]
    except (ValueError, KeyError, TypeError) as exc:
        raise UpdateError(f"the update manifest is not readable ({exc})") from exc
    if not public:
        raise UpdateError("this build has no update key: updates are off")
    if not any(ed25519.verify(k, message, s) for k in public for s in sigs):
        raise UpdateError("the update manifest is not signed by us: ignored")
    try:  # signed by us from here on; still parsed strictly
        p = json.loads(payload_text)
        if p["format"] != FORMAT or p["product"] != PRODUCT:
            raise ValueError("another format or product")
        serial = int(p["serial"])
        app, models, heads = p.get("app") or {}, p.get("models") or {}, p.get("heads") or {}
        update = Update(
            serial=serial,
            app_version=str(app.get("version", "")),
            app_url=str(app.get("url", "")),
            notes={"ar": str(app.get("notes_ar", "")), "en": str(app.get("notes_en", ""))},
            models_version=str(models.get("version", "")),
            model_files={str(n): dict(m) for n, m in (models.get("files") or {}).items()},
            heads_version=int(heads.get("version", 0)),
            head_files={str(n): dict(m) for n, m in (heads.get("files") or {}).items()},
        )
        for url in [update.app_url, *(str(m.get("url", "")) for m in (*update.model_files.values(), *update.head_files.values()))]:
            if url:
                fetch.check_url(url)
    except (ValueError, KeyError, TypeError, AttributeError, fetch.DownloadError) as exc:
        raise UpdateError(f"the update manifest is not usable ({exc})") from exc
    if serial < min_serial:
        raise UpdateError(f"the update manifest is older (serial {serial}) than one already seen ({min_serial}): ignored")
    return update


def check(min_serial: int = 0, fetcher: Callable[[str], bytes] | None = None) -> Update | None:
    """Fetch and verify the manifest. None when updates are off in this build. UpdateError when it cannot be
    fetched or trusted (the caller carries on with what is installed)."""
    if not release.UPDATE_MANIFEST_URL or not release.UPDATE_PUBLIC_KEYS:
        return None
    try:
        raw = (fetcher or (lambda url: fetch.fetch_small(url, MAX_MANIFEST_BYTES)))(release.UPDATE_MANIFEST_URL)
    except fetch.DownloadError as exc:
        raise UpdateError(str(exc)) from exc
    return parse_signed(raw, release.UPDATE_PUBLIC_KEYS, min_serial)


def heads_newer(update: Update, installed_version: int) -> bool:
    return bool(update.head_files) and update.heads_version > installed_version


def apply_data(
    update: Update,
    *,
    models: Path,
    heads_folder: Path,
    downloads: Path,
    installed_heads_version: int,
    progress: modelpack.Progress | None = None,
    cancel: threading.Event | None = None,
    allow_http: bool = False,
) -> list[str]:
    """Bring the models and the heads to what the (verified) manifest names. Each part is checked and swapped in
    on its own, the previous one kept (``models.prev`` / ``heads.prev``). Returns what was changed
    (``"models"`` / ``"heads"``). Raises UpdateError; whatever was not reached stays as it was."""
    changed = []
    try:
        if models_differ(update, modelpack.current_shas(models)):
            modelpack.install_files(update.model_files, models, downloads, progress, cancel, allow_http)
            changed.append("models")
        if heads_newer(update, installed_heads_version):
            heads.install_files(update.head_files, update.heads_version, heads_folder, downloads, cancel, allow_http)
            changed.append("heads")
    except (modelpack.PackError, heads.HeadsError) as exc:
        raise UpdateError(str(exc)) from exc
    return changed


def models_differ(update: Update, installed: dict[str, str]) -> bool:
    """Does the manifest name model files other than the installed ones (``installed``: name -> sha256)?"""
    wanted = {n: str(m.get("sha256", "")) for n, m in update.model_files.items()}
    return bool(wanted) and wanted != installed
