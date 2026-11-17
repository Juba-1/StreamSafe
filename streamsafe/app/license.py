"""The streamer's subscription: a Dodo Payments licence key.

Bought on Dodo's checkout -> Dodo e-mails the key -> pasted once in the window -> ``activate`` binds it to this PC
(an "instance"; the product's activation limit says how many PCs) -> ``check`` before every start and every few
hours while protection runs. Dodo itself switches the key off when the subscription is cancelled, paused or not
paid, so a lapsed subscription stops working without anything else to do here.

Only Dodo's PUBLIC licence endpoints are used (no API key in the app). The answer is fail-closed: an explicit "not
valid" removes the stored licence at once; only a network failure is forgiven, and only for
``release.LICENSE_GRACE_DAYS`` after the last good check. The plan (normal / Pro) comes from the product the key
belongs to (``release.PRODUCT_PLANS``).

A desktop program can always be cracked by someone determined; keys checked against the server stop sharing and
lapsed subscriptions, which is what matters for a subscription.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import platform
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

from streamsafe.app import paths, release

TIMEOUT_S = 15
Post = Callable[[str, dict], tuple[int, dict]]  # (url, body) -> (HTTP status, JSON); status 0 = no network


@dataclass
class License:
    key: str
    instance_id: str
    product_id: str
    plan: str  # "basic" / "pro"
    mode: str  # "test" / "live"
    checked_at: str  # ISO time of the last good answer from Dodo
    customer_email: str = ""


@dataclass(frozen=True)
class Status:
    ok: bool
    reason: str = ""  # "" / no_license / invalid / limit / not_found / inactive / offline_too_long / network / unknown_product
    plan: str = "basic"
    grace: bool = False  # ok only because the network is down (within the grace days)


def _file(env: Mapping[str, str] | None = None):
    return paths.data_dir(dict(env) if env is not None else None) / "license.json"


def load(env: Mapping[str, str] | None = None) -> License | None:
    try:
        raw = json.loads(_file(env).read_text(encoding="utf-8"))
        return License(**{k: raw[k] for k in License.__dataclass_fields__ if k in raw})  # type: ignore[arg-type]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save(lic: License, env: Mapping[str, str] | None = None) -> None:
    f = _file(env)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(asdict(lic), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, f)


def forget(env: Mapping[str, str] | None = None) -> None:
    with contextlib.suppress(OSError):
        _file(env).unlink()


def http_post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        with contextlib.suppress(Exception):
            return exc.code, json.loads(exc.read() or b"{}")
        return exc.code, {}
    except (urllib.error.URLError, OSError, ValueError):
        return 0, {}


def _url(path: str, mode: str | None = None) -> str:
    return release.DODO_BASE[mode or release.DODO_MODE] + path


def device_name() -> str:
    return f"{platform.node() or 'PC'} ({platform.system()})"[:100]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def activate(key: str, env: Mapping[str, str] | None = None, post: Post = http_post, now: Callable[[], dt.datetime] = _now) -> Status:
    """Bind ``key`` to this PC and remember it. The key's spaces from a paste are dropped."""
    key = "".join(key.split())
    if not key:
        return Status(False, "no_license")
    code, data = post(_url("/licenses/activate"), {"license_key": key, "name": device_name()})
    if code == 0:
        return Status(False, "network")
    if code not in (200, 201):
        return Status(False, {403: "inactive", 404: "not_found", 422: "limit"}.get(code, "invalid"))
    product = str((data.get("product") or {}).get("product_id") or "")
    plan = release.PRODUCT_PLANS.get(product, "")
    if release.PRODUCT_PLANS and not plan:
        return Status(False, "unknown_product")  # a key of another product of the same Dodo account
    lic = License(key, str(data.get("id") or ""), product, plan or "basic", release.DODO_MODE, now().isoformat(),
                  str((data.get("customer") or {}).get("email") or ""))  # fmt: skip
    save(lic, env)
    return Status(True, plan=lic.plan)


def check(env: Mapping[str, str] | None = None, post: Post = http_post, now: Callable[[], dt.datetime] = _now) -> Status:
    """Is this PC allowed to protect a stream now? Asks Dodo; forgives only a network failure, for a few days."""
    lic = load(env)
    if lic is None:
        return Status(False, "no_license")
    code, data = post(_url("/licenses/validate", lic.mode), {"license_key": lic.key, "license_key_instance_id": lic.instance_id})
    if code == 200 and data.get("valid") is True:
        lic.checked_at = now().isoformat()
        save(lic, env)
        return Status(True, plan=lic.plan)
    if code in (0, 429) or code >= 500:  # Dodo not reachable: the last good answer holds for the grace days
        try:
            age = now() - dt.datetime.fromisoformat(lic.checked_at)
        except ValueError:
            age = dt.timedelta(days=9999)
        if age <= dt.timedelta(days=release.LICENSE_GRACE_DAYS):
            return Status(True, "network", plan=lic.plan, grace=True)
        return Status(False, "offline_too_long", plan=lic.plan)
    forget(env)  # an explicit "no" (cancelled, unpaid, deactivated elsewhere): nothing kept to be forgiven
    return Status(False, "invalid")


def deactivate(env: Mapping[str, str] | None = None, post: Post = http_post) -> bool:
    """Free this PC's activation (moving to another PC). The stored licence goes either way."""
    lic = load(env)
    if lic is None:
        return True
    code, _ = post(_url("/licenses/deactivate", lic.mode), {"license_key": lic.key, "license_key_instance_id": lic.instance_id})
    forget(env)
    return code in (200, 204)


def masked(key: str) -> str:
    return key[:4] + "…" + key[-4:] if len(key) > 10 else "…"
