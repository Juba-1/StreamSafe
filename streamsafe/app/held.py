"""The held engine: loaded ahead, waiting for Start.

Loading the models takes seconds to minutes (TensorRT sessions, a first DirectML run). The window keeps one engine
loaded while nothing runs, started with exactly what Start would run now (``Controller.prewarm``); its run waits
for "go" on standard input (``engine.wait_for_go``). Start then sends "go" (``take``) and the protection runs
at once. Anything that would change the run (a setting, the key, the port) makes the held engine stale: it is
ended and a new one loads after the change has settled. When it ends by itself (it could not load), a new one is
tried only after ``RETRY_S``. Closing the window, or the window itself gone, ends it (standard input closes).

Only one process at a time: a held engine never runs beside a streaming one.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from streamsafe.app import engine

SETTLE_S = 3.0  # a change must stay this long before a new engine loads (the slider, typing a key)
RETRY_S = 60.0  # after a held engine ended by itself
KEEP_LINES = 240

IGNORED = ("THR_SCALE",)  # read live from the control file: changing it does not need another engine


def spec_of(cmd: list[str], env: Mapping[str, str]) -> tuple:
    return tuple(cmd), tuple(sorted((k, v) for k, v in env.items() if k not in IGNORED))


class Held:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._proc: engine.EngineProcess | None = None
        self._spec: tuple = ()
        self._lines: list[str] = []
        self._wanted: tuple = ()
        self._wanted_since = 0.0
        self._failed_at = -1e9

    @property
    def ready(self) -> bool:
        """A held engine is running and has said its models are loaded."""
        with self._lock:
            return self._proc is not None and self._proc.running and any("protection_ready" in x for x in self._lines)

    @property
    def loading(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.running and not any("protection_ready" in x for x in self._lines)

    def want(self, cmd: list[str], env: Mapping[str, str], secrets: tuple[str, ...], new_log: Callable[[], Path]) -> None:
        """Make sure an engine for (cmd, env) is held: start one, or replace a stale one once the change settled."""
        spec = spec_of(cmd, env)
        now = self._clock()
        with self._lock:
            if spec != self._wanted:
                self._wanted, self._wanted_since = spec, now
            proc = self._proc
            if proc is not None and proc.running and self._spec == spec:
                return
            if proc is not None and not proc.running:
                if not proc.stop_requested:
                    self._failed_at = now  # ended by itself: it could not load; try again later
                self._proc = None
                proc = None
            if now - self._wanted_since < SETTLE_S or now - self._failed_at < RETRY_S:
                return
            old, self._proc, self._spec, self._lines = proc, None, spec, []
            new = engine.EngineProcess(self._line, engine.Redactor(*secrets))
            try:
                new.start(cmd, {**env, engine.HOLD_ENV: "1"}, new_log())
            except OSError:
                self._failed_at = now
                return
            self._proc = new
        if old is not None:
            _end(old)

    def _line(self, line: str) -> None:
        with self._lock:
            self._lines = [*self._lines[-KEEP_LINES:], line]

    def take(self, cmd: list[str], env: Mapping[str, str]) -> tuple[engine.EngineProcess, list[str]] | None:
        """(the held engine, what it said while held) if it runs exactly (cmd, env): it is no longer held, and its
        loading lines belong in the run's log and report. Otherwise None, and any held engine is ended (Start
        runs its own)."""
        with self._lock:
            proc, lines, self._proc = self._proc, list(self._lines), None
            if proc is not None and proc.running and self._spec == spec_of(cmd, env):
                return proc, lines
        if proc is not None:
            _end(proc, wait=True)
        return None

    def drop(self, wait: bool = False) -> None:
        with self._lock:
            proc, self._proc, self._wanted = self._proc, None, ()
        if proc is not None:
            _end(proc, wait)


def _end(proc: engine.EngineProcess, wait: bool = False) -> None:
    """End a held engine: standard input closes and it exits (it opened nothing). Off the caller's thread unless
    ``wait``: the window's tick never waits for it."""
    if wait:
        proc.stop(wait_s=10)
    else:
        threading.Thread(target=proc.stop, kwargs={"wait_s": 10}, name="held-end", daemon=True).start()
