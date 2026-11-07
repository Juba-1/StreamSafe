"""Link to OBS: a tiny state file read by the OBS guard script (``obs/streamsafe_guard.lua``).

One line, rewritten atomically: ``<alive> <seq> <muted> <full>``
  alive  1 while the engine runs, 0 after a clean stop
  seq    output frame counter: if it stops moving, the engine is hung or dead
  muted  game audio must be muted now (whole-screen cover, see runtime/gate.py)
  full   the frame on screen is full-screen (scene card / fail-closed slate)
The script mutes the game audio when ``muted`` is 1 AND whenever the engine stops (alive 0, file missing,
or seq frozen), and then also shows its own slate over the game: OBS stays fail closed even when this
process crashes. No network, no password: a file in the user's own folder.
"""

from __future__ import annotations

import os
import time

MIN_PERIOD_S = 0.1  # unchanged state is rewritten at most 10x per second (the heartbeat)


class ObsStateWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self._tmp = path + ".tmp"
        self._last: tuple[int, int] | None = None
        self._at = 0.0
        self.errors = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def update(self, seq: int, muted: bool, full: bool) -> None:
        state = (int(muted), int(full))
        now = time.monotonic()
        if state == self._last and now - self._at < MIN_PERIOD_S:
            return
        if self._write(f"1 {seq} {state[0]} {state[1]}\n"):
            self._last, self._at = state, now

    def close(self) -> None:
        self._write("0 -1 1 1\n")

    def _write(self, line: str) -> bool:
        try:
            with open(self._tmp, "w") as f:
                f.write(line)
            os.replace(self._tmp, self.path)  # readers never see a half-written line
            return True
        except OSError:  # Windows: the reader had the file open this instant; next frame retries
            self.errors += 1
            return False
