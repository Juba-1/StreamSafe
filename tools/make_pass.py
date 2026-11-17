"""Make a temporary Pro pass (see streamsafe/app/passkey.py).

PASS_SIGNING_KEY=<hex 32-byte ed25519 private key> python tools/make_pass.py --who "name or email" [--days 30]
python tools/make_pass.py --new-keypair      (prints a new private key to keep secret, and its public key)
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import secrets

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat


def b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", default="")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--plan", default="pro")
    ap.add_argument("--new-keypair", action="store_true")
    a = ap.parse_args()
    if a.new_keypair:
        k = Ed25519PrivateKey.generate()
        print("PRIVATE", k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex())
        print("PUBLIC", k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex())
        return
    k = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(os.environ["PASS_SIGNING_KEY"].strip()))
    last = dt.date.today() + dt.timedelta(days=a.days)
    fields = {"p": a.plan, "e": last.isoformat(), "w": a.who.strip()[:60], "i": secrets.token_hex(3)}
    note = json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode()
    print(f"SSP-{b64(note)}.{b64(k.sign(note))}")
    print(f"works until {last.isoformat()} for {a.who.strip()[:60]!r}")


if __name__ == "__main__":
    main()
