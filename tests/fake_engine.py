"""Stands in for ``<exe> run ...`` in the controller tests: writes the OBS state file like the engine does and ends
when its standard input closes (the window's Stop). It prints the output address it was given so the test can see
that the window's redaction works."""

import os
import sys
import time
from pathlib import Path

state = Path(os.environ["OBS_STATE"])
state.parent.mkdir(parents=True, exist_ok=True)
print("args:", " ".join(sys.argv[1:]), flush=True)
print("output:", os.environ.get("STREAMSAFE_OUTPUT_URL", "-"), flush=True)
for line in filter(None, os.environ.get("FAKE_STATUS", "").split("|")):  # STATUS lines the real engine would print
    print("STATUS " + line, flush=True)
if os.environ.get("FAKE_EXIT"):
    sys.exit(int(os.environ["FAKE_EXIT"]))
if os.environ.get("FAKE_CRASH"):
    sys.exit(3)
seq = 0
import threading  # noqa: E402


def watch():
    sys.stdin.read()
    os._exit(0)


threading.Thread(target=watch, daemon=True).start()
while True:
    seq += 1
    full = 1 if os.environ.get("FAKE_FULL") else 0
    state.write_text(f"1 {seq} 0 {full}", encoding="ascii")
    time.sleep(0.05)
