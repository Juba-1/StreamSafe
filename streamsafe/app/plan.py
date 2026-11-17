"""What the streamer's subscription allows.

Two plans (Ali, 17 Nov 2026): the normal one - full protection, the covers carry StreamSafe's logo - and Pro,
which may also put the streamer's / a sponsor's logo on the covers. The plan comes from the licence the app
activated (``license.json`` in the data folder, written by the licence check); until a licence system is
connected every copy is on the normal plan. ``STREAMSAFE_PLAN`` overrides it for tests and support.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

from streamsafe.app import paths

BASIC, PRO = "basic", "pro"
FEATURES = {"custom_brand": PRO}  # feature -> the lowest plan that has it
_ORDER = (BASIC, PRO)


def current(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    forced = env.get("STREAMSAFE_PLAN", "")
    if forced in _ORDER:
        return forced
    try:
        data = json.loads((paths.data_dir(dict(env)) / "license.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return BASIC
    got = data.get("plan") if isinstance(data, dict) else None
    return got if got in _ORDER else BASIC


def allows(feature: str, env: Mapping[str, str] | None = None) -> bool:
    need = FEATURES.get(feature, BASIC)
    return _ORDER.index(current(env)) >= _ORDER.index(need)
