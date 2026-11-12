"""Downloads: HTTPS only, to a ``.part`` file that is resumed after a broken connection, and nothing is kept
unless its sha256 is the pinned one. Used for the model pack, the update manifest's files and NVIDIA's wheels."""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

CHUNK = 1 << 20
TIMEOUT_S = 30
USER_AGENT = "StreamSafe-app"  # no version, no machine or user detail

Progress = Callable[[int, int], None]  # (bytes so far, total bytes or 0 when unknown)


class DownloadError(Exception):
    """The file could not be downloaded (network, server, disk). ``status``: the HTTP status when the server
    answered with one (401 / 403: private or no access, 404: not there), else 0."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class IntegrityError(DownloadError):
    """The bytes received are not the pinned ones (sha256 or size): they were deleted."""


class CancelledError(DownloadError):
    pass


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, getattr(req, "streamsafe_allow_http", False))
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.streamsafe_allow_http = getattr(req, "streamsafe_allow_http", False)  # type: ignore[attr-defined]
        return new


def check_url(url: str, allow_http: bool = False) -> None:
    """Refuse everything but https:// (tests may allow http:// to this machine only)."""
    parts = urlsplit(url)
    local = parts.hostname in ("127.0.0.1", "localhost")
    if parts.scheme == "https" or (allow_http and parts.scheme == "http" and local):
        return
    raise DownloadError(f"refused address (https only): {parts.scheme}://{parts.hostname}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _open(url: str, start: int, allow_http: bool):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **({"Range": f"bytes={start}-"} if start else {})})
    req.streamsafe_allow_http = allow_http  # type: ignore[attr-defined]
    return urllib.request.build_opener(_HttpsOnlyRedirects).open(req, timeout=TIMEOUT_S)


def download(
    url: str,
    dest: Path,
    sha256: str,
    size: int = 0,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    allow_http: bool = False,
    max_bytes: int = 0,
) -> Path:
    """Download ``url`` to ``dest``; returns ``dest`` only when its sha256 is ``sha256`` (required).
    A ``dest.part`` left by an earlier try is continued. ``size`` (when known) and ``max_bytes`` bound the transfer."""
    if not sha256 or len(sha256) != 64:
        raise IntegrityError("no sha256 is pinned for this download: refused")
    check_url(url, allow_http)
    dest.parent.mkdir(parents=True, exist_ok=True)
    limit = size or max_bytes
    if dest.exists():
        if sha256_file(dest) == sha256.lower():
            return dest
        dest.unlink()
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if limit and have > limit:
        part.unlink()
        have = 0
    try:
        if not (size and have == size):
            try:
                resp = _open(url, have, allow_http)
            except urllib.error.HTTPError as exc:
                if exc.code != 416 or not have:  # 416: the part file is already the whole file (or stale)
                    raise
                part.unlink()
                have = 0
                resp = _open(url, 0, allow_http)
            with resp:
                if have and resp.status != 206:  # the server ignored the range: start again
                    have = 0
                total = size or (have + int(resp.headers.get("Content-Length") or 0))
                with open(part, "ab" if have else "wb") as f:
                    while chunk := resp.read(CHUNK):
                        if cancel is not None and cancel.is_set():
                            raise CancelledError("cancelled")
                        have += len(chunk)
                        if limit and have > limit:
                            raise IntegrityError("the server sent more bytes than the pinned size")
                        f.write(chunk)
                        if progress:
                            progress(have, total)
    except CancelledError:
        raise  # the part file stays: the next try continues it
    except IntegrityError:
        part.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:  # URLError and HTTPError are OSErrors
        raise DownloadError(f"download failed: {_reason(exc)}", _status(exc)) from exc
    if (size and have != size) or sha256_file(part) != sha256.lower():
        part.unlink(missing_ok=True)
        raise IntegrityError("the downloaded file does not match its pinned sha256: deleted")
    os.replace(part, dest)
    return dest


def fetch_small(url: str, max_bytes: int = 1 << 20, allow_http: bool = False) -> bytes:
    """A small file into memory (the update manifest). Its content is verified by the caller (signature)."""
    check_url(url, allow_http)
    try:
        with _open(url, 0, allow_http) as resp:
            data = resp.read(max_bytes + 1)
    except (OSError, ValueError) as exc:
        raise DownloadError(f"download failed: {_reason(exc)}", _status(exc)) from exc
    if len(data) > max_bytes:
        raise DownloadError("the file is larger than expected")
    return data


def _status(exc: BaseException) -> int:
    return exc.code if isinstance(exc, urllib.error.HTTPError) else 0


def _reason(exc: BaseException) -> str:
    """The error without the address's query string or path (they may carry a token)."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    reason = getattr(exc, "reason", None)
    return str(reason if reason is not None else exc)[:200]
