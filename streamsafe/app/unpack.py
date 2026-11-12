"""Unpacking a zip (the model pack, a vendor wheel) without trusting it: no path can leave the target folder,
links are refused, and the unpacked size is bounded by what is really written, not by what the zip claims."""

from __future__ import annotations

import stat
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

CHUNK = 1 << 20

Progress = Callable[[int, int], None]


class UnsafeArchiveError(Exception):
    """The archive holds something that must not be unpacked (a path outside the folder, a link, too much data)."""


def safe_relpath(name: str) -> PurePosixPath:
    """``name`` as a relative path that stays inside the folder it is unpacked into, or UnsafeArchiveError."""
    if not name or "\\" in name or "\x00" in name or ":" in name or name.startswith("/"):
        raise UnsafeArchiveError(f"unsafe path in archive: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise UnsafeArchiveError(f"unsafe path in archive: {name!r}")
    return path


def extract(
    archive: Path,
    dest: Path,
    choose: Callable[[str], str | None],
    max_bytes: int,
    max_files: int = 20000,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> list[str]:
    """Unpack the members ``choose`` maps to a relative path (None = skip) into ``dest``. Returns those paths."""
    written, done = [], 0
    try:
        with zipfile.ZipFile(archive) as zf:
            wanted = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                target = choose(info.filename)
                if target is None:
                    continue
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise UnsafeArchiveError(f"link in archive: {info.filename!r}")
                wanted.append((info, safe_relpath(target)))
            total = sum(info.file_size for info, _ in wanted)
            if len(wanted) > max_files or total > max_bytes:
                raise UnsafeArchiveError("the archive is larger than anything it should hold")
            if len({str(rel).lower() for _, rel in wanted}) != len(wanted):
                raise UnsafeArchiveError("the archive holds the same file twice")
            for info, rel in wanted:
                out = dest.joinpath(*rel.parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(out, "wb") as dst:
                    while chunk := src.read(CHUNK):
                        if cancel is not None and cancel.is_set():
                            raise InterruptedError("cancelled")
                        done += len(chunk)
                        if done > max_bytes:
                            raise UnsafeArchiveError("the archive unpacks to more than it declares")
                        dst.write(chunk)
                        if progress:
                            progress(done, total)
                written.append(rel.as_posix())
    except zipfile.BadZipFile as exc:
        raise UnsafeArchiveError(f"not a readable zip file ({exc})") from exc
    return written
