"""Windows probe: does starting a process hang while another thread waits on standard input (a pipe)?
Run with stdin as a pipe nobody writes to. MODE=plain: as the engine did; MODE=fixed: with engine.detach_stdin."""

import multiprocessing as mp
import os
import sys
import threading
import time


def child(q):
    q.put("hello")


def main() -> None:
    mode = os.environ.get("MODE", "plain")
    if mode == "fixed":
        sys.path.insert(0, os.getcwd())
        from streamsafe.app.engine import detach_stdin

        stream = detach_stdin()
    else:
        stream = sys.stdin
    threading.Thread(target=lambda: stream.readline(), daemon=True).start()
    time.sleep(1)  # the reader is now blocked in its read
    q = mp.Queue()
    t = time.monotonic()
    p = mp.Process(target=child, args=(q,), daemon=True)
    done = threading.Event()

    def go():
        p.start()
        done.set()

    threading.Thread(target=go, daemon=True).start()
    ok = done.wait(20)
    msg = q.get(timeout=20) if ok else None
    print(f"PROBE mode={mode} started={ok} in {time.monotonic() - t:.1f}s child_said={msg}", flush=True)
    os._exit(0 if ok else 3)


if __name__ == "__main__":
    main()
