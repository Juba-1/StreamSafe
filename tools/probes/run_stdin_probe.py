"""Starts stdin_spawn.py with a pipe as its stdin that is never written, prints its answer."""

import os
import subprocess
import sys

results = {}
for mode in ("plain", "fixed"):
    p = subprocess.Popen([sys.executable, "tools/probes/stdin_spawn.py"], stdin=subprocess.PIPE, env={**os.environ, "MODE": mode})
    try:
        code = p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()
        code = "timeout"
    print(f"RESULT mode={mode} exit={code}", flush=True)
    results[mode] = code
sys.exit(0 if results["fixed"] == 0 else 1)  # the fix must start the process; "plain" shows the old behaviour
