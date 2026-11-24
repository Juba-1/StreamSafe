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
