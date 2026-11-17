"""What the streamer's subscription allows.

Two plans (Ali, 17 Nov 2026): the normal one - full protection, the covers carry StreamSafe's logo - and Pro,
which may also put the streamer's / a sponsor's logo on the covers. The plan comes from the licence the app
activated (``license.json`` in the data folder, written by the licence check); until a licence system is
connected every copy is on the normal plan. ``STREAMSAFE_PLAN`` overrides it for tests and support.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

BASIC, PRO = "basic", "pro"
FEATURES = {"custom_brand": PRO}  # feature -> the lowest plan that has it
_ORDER = (BASIC, PRO)


def current(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    forced = env.get("STREAMSAFE_PLAN", "")
    if forced in _ORDER:
        return forced
    from streamsafe.app import license

    lic = license.load(env)  # written only by license.activate (the product the key was bought for)
    if lic is not None and lic.plan in _ORDER:
        return lic.plan
    from streamsafe.app import passkey

    held = passkey.active(env)  # a supporter's pass: Pro until its last day, then back to normal
    return held.plan if held is not None and held.plan in _ORDER else BASIC


def allows(feature: str, env: Mapping[str, str] | None = None) -> bool:
    need = FEATURES.get(feature, BASIC)
    return _ORDER.index(current(env)) >= _ORDER.index(need)
