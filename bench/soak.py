"""Long-run (soak) test of the live path, as close to a stream as a server allows.

  sender : FFmpeg plays the clips in a loop in REAL TIME (-re) as H.264 over UDP, like OBS / a capture feed
  engine : python -m streamsafe, live UDP input -> realtime mode -> H.264 UDP output (the streaming encoder),
           with the OBS state file on (heartbeat + mute)
  watcher: every 100 ms reads the state file the way the OBS guard script does (largest heartbeat gap =
           how close OBS came to failing closed); every 10 s the engine's memory (RSS of its processes) and
           GPU memory -> leaks show up as a rising line.

usage: python -m bench.soak OUT_DIR MINUTES CLIP [CLIP ...]      (needs ffmpeg on PATH, psutil)
Writes OUT_DIR/soak.json + stats.json; prints "SOAK <json>".
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

IN_URL = "udp://127.0.0.1:9000"
OUT_URL = "udp://127.0.0.1:9100?pkt_size=1316"
SAMPLE_S = 10.0


def playlist(out: Path, clips: list[str], ff: str) -> Path:
    """The clips as one 1080p30 H.264 file (same format throughout, so it loops cleanly)."""
    parts = []
    for i, c in enumerate(clips):
        p = out / f"part{i}.mp4"
        subprocess.run([ff, "-loglevel", "error", "-y", "-i", c, "-an", "-vf", "scale=1920:1080,fps=30",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", p], check=True)  # fmt: skip
        parts.append(p)
    lst = out / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    joined = out / "playlist.mp4"
    subprocess.run([ff, "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined], check=True)
    for p in parts:
        p.unlink()
    return joined


def gpu_mb() -> float:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return float(q.stdout.split()[0])
    except (OSError, ValueError, IndexError):
        return -1.0


def tree_rss_mb(proc: psutil.Process) -> float:
    total = 0
    with contextlib.suppress(psutil.Error):
        for p in [proc, *proc.children(recursive=True)]:
            with contextlib.suppress(psutil.Error):
                total += p.memory_info().rss
    return total / 2**20


class HeartbeatWatch(threading.Thread):
    """Reads the state file like obs/streamsafe_guard.lua (100 ms tick); records the longest frozen time."""

    def __init__(self, path: Path) -> None:
        super().__init__(daemon=True)
        self.path, self.stop = path, threading.Event()
        self.max_gap_ms, self.stale_events, self.reads, self.muted_ticks = 0.0, 0, 0, 0
        self._seq, self._since = None, None

    def run(self) -> None:
        while not self.stop.wait(0.1):
            try:
                alive, seq, muted, _ = self.path.read_text().split()
            except (OSError, ValueError):
                continue
            if alive != "1":
                continue
            self.reads += 1
            self.muted_ticks += muted == "1"
            now = time.monotonic()
            if seq != self._seq:
                if self._since is not None:
                    gap = 1000 * (now - self._since)
                    self.max_gap_ms = max(self.max_gap_ms, gap)
                    self.stale_events += gap >= 500
                self._seq, self._since = seq, now


def main(argv: list[str]) -> None:
    out, minutes, clips = Path(argv[0]), float(argv[1]), argv[2:]
    out.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    assert ff, "ffmpeg must be on PATH"
    src = playlist(out, clips, ff)
    state = out / "state.txt"
    env = dict(os.environ, OBS_STATE=str(state))
    log = open(out / "engine.log", "w")  # noqa: SIM115 - open for the engine's whole life
    engine = subprocess.Popen(
        [sys.executable, "-m", "streamsafe", "--input", f"{IN_URL}?fifo_size=1000000&overrun_nonfatal=1",
         "--output", OUT_URL, "--mode", "realtime", "--stats", str(out / "stats.json")],
        env=env, stdout=log, stderr=subprocess.STDOUT,
    )  # fmt: skip
    eng = psutil.Process(engine.pid)
    time.sleep(90)  # models load; frames sent now would only be dropped
    sender = subprocess.Popen(
        [ff, "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", src, "-t", str(minutes * 60),
         "-c", "copy", "-f", "mpegts", f"{IN_URL}?pkt_size=1316"]
    )  # fmt: skip
    watch = HeartbeatWatch(state)
    watch.start()
    series, t0 = [], time.monotonic()
    while sender.poll() is None and engine.poll() is None:
        time.sleep(SAMPLE_S)
        series.append([round(time.monotonic() - t0), round(tree_rss_mb(eng)), gpu_mb()])
        if len(series) % 30 == 0:
            print("SOAK_TICK", json.dumps(series[-1]), flush=True)
    sender.wait()
    try:
        engine.wait(timeout=120)  # input ended -> the engine finishes and writes its stats
    except subprocess.TimeoutExpired:
        engine.kill()
    watch.stop.set()
    log.close()
    rss = [s[1] for s in series] or [0]
    k = max(1, len(rss) // 10)
    early, late = sum(rss[:k]) / k, sum(rss[-k:]) / k  # first / last tenth of the run
    report = dict(
        minutes=minutes,
        engine_exit=engine.returncode,
        rss_mb_first10pct=round(early),
        rss_mb_last10pct=round(late),
        rss_growth_mb_per_h=round((late - early) / max(1e-6, (series[-1][0] if series else 1) / 3600), 1),
        gpu_mb_max=max((s[2] for s in series), default=-1),
        heartbeat_max_gap_ms=round(watch.max_gap_ms),
        heartbeat_stale_events=watch.stale_events,
        heartbeat_reads=watch.reads,
        series=series,
    )
    stats = out / "stats.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        late_frames = s.get("late_frames", {})
        report["engine"] = s["summary"] | {"late_share": late_frames.get("mean"), "render_ms_p95": s.get("render_ms", {}).get("p95")}
    (out / "soak.json").write_text(json.dumps(report, indent=1))
    print("SOAK " + json.dumps({k: v for k, v in report.items() if k != "series"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
