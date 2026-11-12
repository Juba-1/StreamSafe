"""Which ONNX Runtime a process of the frozen app loads: the bundled DirectML build (default), or the NVIDIA
build that ``nvidia_pack`` unpacked into the user's data folder.

The choice is made by whoever starts the process, through ``STREAMSAFE_RUNTIME=nvidia`` in its environment, and
it must take effect before ``onnxruntime`` is imported anywhere in that process. So ``activate`` is called from
the frozen build's start-up hook (``tools/pyi_rth_streamsafe.py``: runs in every process of the exe, also in the
analyzer's worker process) and again at the top of ``entry.main``.

It only ever points at the one folder ``nvidia_pack`` sealed (``READY.json``, built from this build's pinned
sha256 list). If that folder is not ready, nothing is changed and the reason is returned: the bundled runtime
stays in use.
"""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping

from streamsafe.app import nvidia_pack, paths, release

RUNTIME_ENV = "STREAMSAFE_RUNTIME"
NVIDIA, DIRECTML = "nvidia", "directml"
_ACTIVE: list[str] = []


def wanted(env: MutableMapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return env.get(RUNTIME_ENV, "").strip().lower()


def active() -> bool:
    return bool(_ACTIVE)


def activate(env: MutableMapping[str, str] | None = None, sys_path: list[str] | None = None) -> str:
    """Switch this process to the NVIDIA runtime when asked. Returns "" when it is now active, otherwise why not
    (also "not asked"). Idempotent."""
    env = os.environ if env is None else env
    sys_path = sys.path if sys_path is None else sys_path
    if wanted(env) != NVIDIA:
        return "not asked"
    if tuple(sys.version_info[:2]) != release.NVIDIA_PYTHON:
        return f"the NVIDIA runtime needs Python {release.NVIDIA_PYTHON[0]}.{release.NVIDIA_PYTHON[1]}"
    nvidia = paths.nvidia_dir(env)
    if not nvidia_pack.ready(nvidia):
        return "the NVIDIA libraries are not installed (or belong to another version of the app)"
    site = str(nvidia_pack.site_dir(nvidia))
    if site in sys_path:
        return ""
    if "onnxruntime" in sys.modules:
        return "ONNX Runtime was already loaded in this process"
    sys_path.insert(0, site)
    dirs = [str(d) for d in nvidia_pack.dll_dirs(nvidia)]
    env["PATH"] = os.pathsep.join([*dirs, env.get("PATH", "")])
    if sys.platform == "win32" and env is os.environ:
        for d in dirs:
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
    _ACTIVE.append(site)
    return ""
