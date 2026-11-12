"""The stream key at rest: protected with Windows DPAPI (``CryptProtectData``), never written as plain text.

DPAPI ties the stored bytes to the Windows user account: another user of the PC, or the file copied to another
PC, cannot read it; no password of ours is involved. Where there is no DPAPI (not Windows) nothing is stored:
``save`` refuses, and the key lives only in the running window's memory.

The key is never logged, never printed and never put on a command line by this app (``engine.py`` hands it to
the engine through the child's environment).
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Protocol

MAGIC = b"SSK1"
MAX_KEY_CHARS = 512
_ENTROPY = b"stream-key-v1"  # not a secret: only keeps these bytes apart from other DPAPI data of the user


class NoSecureStoreError(Exception):
    """This system has no store we trust with a stream key; nothing was written."""


class Protector(Protocol):
    def protect(self, data: bytes) -> bytes: ...

    def unprotect(self, blob: bytes) -> bytes: ...


class Dpapi:
    """Windows Data Protection API, current-user scope, no prompt."""

    UI_FORBIDDEN = 0x1

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class Blob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        self._ctypes, self._blob = ctypes, Blob
        self._crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)  # type: ignore[attr-defined]
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        args = [ctypes.POINTER(Blob), wintypes.LPCWSTR, ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
        self._crypt32.CryptProtectData.argtypes = [*args, ctypes.POINTER(Blob)]
        self._crypt32.CryptProtectData.restype = wintypes.BOOL
        args[1] = ctypes.c_void_p  # the description comes back; not wanted
        self._crypt32.CryptUnprotectData.argtypes = [*args, ctypes.POINTER(Blob)]
        self._crypt32.CryptUnprotectData.restype = wintypes.BOOL
        self._kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        self._kernel32.LocalFree.restype = ctypes.c_void_p

    def _call(self, fn, data: bytes) -> bytes:
        ctypes = self._ctypes
        buf_in, buf_entropy = ctypes.create_string_buffer(data, len(data)), ctypes.create_string_buffer(_ENTROPY, len(_ENTROPY))
        blob_in = self._blob(len(data), ctypes.cast(buf_in, ctypes.POINTER(ctypes.c_char)))
        entropy = self._blob(len(_ENTROPY), ctypes.cast(buf_entropy, ctypes.POINTER(ctypes.c_char)))
        out = self._blob()
        if not fn(ctypes.byref(blob_in), None, ctypes.byref(entropy), None, None, self.UI_FORBIDDEN, ctypes.byref(out)):
            raise OSError(ctypes.get_last_error(), "Windows could not protect / unprotect the data")
        try:
            return ctypes.string_at(out.pbData, out.cbData)
        finally:
            self._kernel32.LocalFree(out.pbData)

    def protect(self, data: bytes) -> bytes:
        return self._call(self._crypt32.CryptProtectData, data)

    def unprotect(self, blob: bytes) -> bytes:
        return self._call(self._crypt32.CryptUnprotectData, blob)


def default_protector() -> Protector | None:
    if sys.platform != "win32":
        return None
    try:
        return Dpapi()
    except (OSError, AttributeError):
        return None


class KeyStore:
    def __init__(self, path: Path, protector: Protector | None = None) -> None:
        self.path = path
        self._protector = protector

    @property
    def available(self) -> bool:
        return self._protector is not None

    def save(self, key: str) -> None:
        """Store ``key`` protected. Raises NoSecureStoreError (nothing written) when there is no protector."""
        if self._protector is None:
            raise NoSecureStoreError("no protected store on this system: the stream key is not saved")
        if not key or len(key) > MAX_KEY_CHARS:
            raise ValueError("empty or over-long stream key")
        blob = MAGIC + self._protector.protect(key.encode("utf-8"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_bytes(blob)
            os.replace(tmp, self.path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()

    def load(self) -> str:
        """The stored key, or "" when there is none or it cannot be read by this user on this PC."""
        if self._protector is None:
            return ""
        try:
            blob = self.path.read_bytes()
            if not blob.startswith(MAGIC):
                return ""
            return self._protector.unprotect(blob[len(MAGIC) :]).decode("utf-8")
        except (OSError, ValueError):
            return ""

    def clear(self) -> None:
        with contextlib.suppress(OSError):
            self.path.unlink()
