"""The Stop button: the engine is told to watch its stdin and ends cleanly at once (not killed after the wait)."""

import os
import sys
import time

from streamsafe.app import engine

CHILD = """
import os, sys, time
from streamsafe.app import engine
assert os.environ.get(engine.STOP_ENV) == "1", "the app did not ask the engine to watch stdin"
engine.stop_on_stdin_close()
print("STATUS {\\"event\\": \\"waiting_obs\\", \\"port\\": 1}", flush=True)
try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt:
    print("CLEAN_STOP", flush=True)
"""


def test_stop_button_ends_the_engine_cleanly_and_quickly(tmp_path):
    lines: list[str] = []
    e = engine.EngineProcess(lines.append)
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    e.start([sys.executable, "-c", CHILD], env)
    end = time.monotonic() + 15
    while time.monotonic() < end and not any("waiting_obs" in line for line in lines):
        time.sleep(0.1)
    t = time.monotonic()
    code = e.stop(wait_s=10)
    assert code == 0 and time.monotonic() - t < 5, (code, lines)
    assert any("CLEAN_STOP" in line for line in lines)


class _SlowWorker:
    def __init__(self, load_s: float, fail: bool = False) -> None:
        self.load_s, self.fail, self.got = load_s, fail, []

    def launch(self) -> None: ...

    def wait_ready(self) -> None:
        import time as _t

        _t.sleep(self.load_s)
        if self.fail:
            raise RuntimeError("analyzer process died while loading its models")

    def submit(self, frame_id, frame, cut) -> None:
        self.got.append(frame_id)

    def signal_stop(self) -> None: ...

    def stop(self, timeout: float = 0) -> None: ...

    def close(self) -> None: ...


def test_models_load_while_the_stream_already_runs():
    """A slow first load (minutes on a fresh PC) no longer blocks the run: frames before it is ready get no
    decisions (the slate), and the status says loading, then ready."""
    import time

    from streamsafe.io import link
    from streamsafe.runtime.runner import _Workers

    said = []
    link._sink = said.append
    try:
        t0 = time.monotonic()
        w = _Workers(lambda: _SlowWorker(0.4))
        assert time.monotonic() - t0 < 0.2, "the run goes on while the models load"
        w.submit(0, None, False)
        assert w.current.got == [] and not w.ready.is_set()
        assert w.ready.wait(3) and w.failed is None
        w.submit(1, None, False)
        assert w.current.got == [1]
        assert any("protection_loading" in s for s in said) and any("protection_ready" in s for s in said)
    finally:
        link._sink = None


def test_a_first_load_that_fails_ends_the_run_with_the_reason():
    from streamsafe.runtime.runner import _Workers

    w = _Workers(lambda: _SlowWorker(0.05, fail=True))
    for _ in range(100):
        if w.failed is not None:
            break
        __import__("time").sleep(0.02)
    assert isinstance(w.failed, RuntimeError) and not w.ready.is_set()
