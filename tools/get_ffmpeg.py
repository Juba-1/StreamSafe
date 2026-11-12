"""Download the pinned LGPL FFmpeg of ``installer/ffmpeg.lock.json``, check its sha256, refuse it if it is a GPL
build, and unpack ``bin/ffmpeg.exe`` + its DLLs + the licence into ``OUT/ffmpeg``.

    python tools/get_ffmpeg.py OUT_DIR [--zip LOCAL.zip]

The sha256 is checked before anything is unpacked. A GPL build (``--enable-gpl`` / ``--enable-nonfree`` in the
build configuration) must never go into the installer: this is checked on the Windows runner by the smoke test
(``ffmpeg -version``), and the lock file is the only place that names the build.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "installer" / "ffmpeg.lock.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    if not url.startswith("https://"):
        raise SystemExit(f"refused: not https: {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as out:  # noqa: S310 - https checked above
        shutil.copyfileobj(resp, out, 1 << 20)


def unpack(archive: Path, out: Path, keep: list[str]) -> list[str]:
    done = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = info.filename.split("/", 1)[1] if "/" in info.filename else info.filename
            if not any(fnmatch.fnmatch(rel, pattern) for pattern in keep) or ".." in rel.split("/"):
                continue
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            done.append(rel)
    return done


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    out = Path(argv[0]) / "ffmpeg"
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    archive = Path(argv[argv.index("--zip") + 1]) if "--zip" in argv else Path(argv[0]) / "ffmpeg-download.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if "--zip" not in argv:
        fetch(lock["url"], archive)
    got = sha256_file(archive)
    if got != lock["sha256"]:
        if "--zip" not in argv:
            archive.unlink(missing_ok=True)
        print(f"FFmpeg sha256 mismatch: got {got}, pinned {lock['sha256']}", file=sys.stderr)
        return 1
    shutil.rmtree(out, ignore_errors=True)
    files = unpack(archive, out, lock["keep"])
    if "bin/ffmpeg.exe" not in files or "LICENSE.txt" not in files:
        print("the FFmpeg archive lacks bin/ffmpeg.exe or LICENSE.txt", file=sys.stderr)
        return 1
    if "--zip" not in argv:
        archive.unlink(missing_ok=True)
    print(f"FFmpeg {lock['name']}: {len(files)} files -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
