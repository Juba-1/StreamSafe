"""Make and sign the update manifest (ed25519). Run on the OWNER's machine; the private key never enters the repository.

    python tools/sign_manifest.py keygen --private KEYFILE          # once: prints the PUBLIC key (hex) for release.py
    python tools/sign_manifest.py sign --private KEYFILE --serial N --out manifest.json \\
        [--app-version 3.0.1 --app-url https://... --notes-ar "..." --notes-en "..."] \\
        [--models-dir DIR --models-url https://host/path/ --models-version 2026-11-20] \\
        [--heads-dir DIR --heads-url https://host/path/ --heads-version 3]

``--serial`` must go up with every manifest (the app ignores a lower one: an old signed file cannot bring old
models back). ``--models-dir`` is the folder written by ``tools/export_onnx.py`` (with its manifest.json);
``--heads-dir`` holds fast_h6.txt, male_h6.txt, thr_h6.txt (and scene_h.txt). Needs ``pip install cryptography``.
The files themselves are uploaded by hand to the URLs; the manifest only says where they are and their sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamsafe.app import updates  # noqa: E402


def _files(folder: Path, base: str, names: list[str]) -> dict[str, dict]:
    out = {}
    for name in names:
        data = (folder / name).read_bytes()
        out[name] = {"url": base.rstrip("/") + "/" + name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return out


def build_payload(args: argparse.Namespace) -> dict:
    payload: dict = {"format": updates.FORMAT, "product": updates.PRODUCT, "serial": args.serial}
    if args.app_version:
        payload["app"] = {
            "version": args.app_version,
            "url": args.app_url or "",
            "notes_ar": args.notes_ar or "",
            "notes_en": args.notes_en or "",
        }
    if args.models_dir:
        folder = Path(args.models_dir)
        listed = list(json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["files"])
        payload["models"] = {"version": args.models_version or "", "files": _files(folder, args.models_url, ["manifest.json", *listed])}
    if args.heads_dir:
        folder = Path(args.heads_dir)
        names = [n for n in ("fast_h6.txt", "male_h6.txt", "thr_h6.txt", "scene_h.txt") if (folder / n).is_file()]
        payload["heads"] = {"version": int(args.heads_version), "files": _files(folder, args.heads_url, names)}
    return payload


def main(argv: list[str] | None = None) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    kg = sub.add_parser("keygen")
    kg.add_argument("--private", required=True)
    sg = sub.add_parser("sign")
    sg.add_argument("--private", required=True)
    sg.add_argument("--serial", type=int, required=True)
    sg.add_argument("--out", required=True)
    for name in (
        "app-version",
        "app-url",
        "notes-ar",
        "notes-en",
        "models-dir",
        "models-url",
        "models-version",
        "heads-dir",
        "heads-url",
        "heads-version",
    ):
        sg.add_argument(f"--{name}")
    args = ap.parse_args(argv)
    raw = serialization.Encoding.Raw, serialization.PublicFormat.Raw
    if args.cmd == "keygen":
        path = Path(args.private)
        if path.exists():
            print(f"{path} exists: not overwritten", file=sys.stderr)
            return 2
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
        print("public key (put it into release.UPDATE_PUBLIC_KEYS):", key.public_key().public_bytes(*raw).hex())
        return 0
    key = serialization.load_pem_private_key(Path(args.private).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        print("not an ed25519 key", file=sys.stderr)
        return 2
    payload_text = json.dumps(build_payload(args), ensure_ascii=False, sort_keys=True)
    envelope = {"payload": payload_text, "signatures": [key.sign(payload_text.encode("utf-8")).hex()]}
    Path(args.out).write_text(json.dumps(envelope, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"signed manifest, serial {args.serial}: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
