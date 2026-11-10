"""Analyzer factories that fail on purpose (top-level so the worker process can import them)."""

from __future__ import annotations

import os

from streamsafe.config import Config
from tests.stubs import stub_analyzer

CRASH_AT = 30


class _Crashing:
    def __init__(self, inner, hard: bool) -> None:
        self.inner, self.hard = inner, hard
        self.models, self.stats, self.diag = inner.models, inner.stats, inner.diag

    def warmup(self) -> None:
        self.inner.warmup()

    def __call__(self, frame_id: int, frame, cut: bool, now=None):
        if frame_id >= CRASH_AT:
            if self.hard:
                os._exit(3)  # the whole analyzer process dies
            raise RuntimeError("model failure")
        return self.inner(frame_id, frame, cut, now)


def process_dies(cfg: Config, scale: float) -> _Crashing:
    return _Crashing(stub_analyzer(cfg, scale), hard=True)


class _DiesOnce(_Crashing):
    """Dies hard at CRASH_AT the first time only (a flag file shared through the environment)."""

    def __call__(self, frame_id: int, frame, cut: bool, now=None):
        flag = os.environ["STREAMSAFE_TEST_FLAG"]
        if frame_id >= CRASH_AT and not os.path.exists(flag):
            open(flag, "w").close()
            os._exit(3)
        return self.inner(frame_id, frame, cut, now)


def process_dies_once(cfg: Config, scale: float) -> _DiesOnce:
    return _DiesOnce(stub_analyzer(cfg, scale), hard=True)
