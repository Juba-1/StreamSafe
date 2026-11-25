"""The held engine on a GPU pod (real models, TensorRT): how long from Start until the protection runs, with the
engine loaded ahead and without it. The platform is a local RTMP receiver (nothing goes out), OBS a local
FFmpeg test picture. Writes $OUT/held.json.  Run from tools/held_job.sh."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ui_live_drive import ffmpeg_obs, pod_runtime

from streamsafe.app.webui import server

OUT = Path(os.environ.get("OUT", "."))
LOG: list[str] = []


def say(text: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    LOG.append(line)


def wait(cond, limit: float, every: float = 0.1) -> float | None:
    """Seconds until ``cond()`` held, or None after ``limit``."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit:
        if cond():
            return round(time.monotonic() - t0, 1)
        time.sleep(every)
    return None


def relay() -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-listen", "1", "-i", "rtmp://127.0.0.1:1935/live/abc123", "-f", "null", "-"]
    )


def one_run(app, label: str) -> dict:
    c = app.ctrl
    rx = relay()
    time.sleep(1)
    t0 = time.monotonic()
    c.start(app.key)
    held = any("HELD ready" in x for x in c._engine_lines)
    prot = wait(lambda: c.link.protection == "ready", 600)
    obs = ffmpeg_obs(["-re", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"],
                     c.settings.srt_port)  # fmt: skip
    connected = wait(lambda: c.link.platform == "connected", 120)
    protected = wait(lambda: app.state()["status"]["key"] in ("protected", "covered"), 60, 0.3)
    total = round(time.monotonic() - t0, 1)
    time.sleep(8)
    loads = [x for x in c._engine_lines if "STREAMSAFE_LOAD" in x or "protection_" in x]
    c.stop()
    obs.terminate()
    rx.terminate()
    obs.wait(10)
    rx.wait(10)
    out = {"label": label, "used_held_engine": held, "protection_ready_after_start_s": prot,
           "platform_connected_after_start_s": total if connected is not None else None, "status_ok_after_obs_s": protected,
           "load_lines": loads, "exit": c.engine.exit_code}  # fmt: skip
    say(json.dumps(out, ensure_ascii=False))
    return out


def main() -> int:
    pod_runtime()
    app = server.WebApp()
    c = app.ctrl
    app.do_platform({"platform": "custom", "rtmp_url": "rtmp://127.0.0.1:1935/live", "key": "abc123"})
    c.run_check()
    say(f"ready: models={c.models_problem() or 'ok'} blockers={c.blockers(app.key)}")
    results = {}
    t0 = time.monotonic()
    results["held_loaded_after_open_s"] = wait(lambda: app.state()["held"] == "ready", 900, 1.0)
    say(f"held engine ready {results['held_loaded_after_open_s']} s after the window opened")
    results["with_held"] = one_run(app, "with held engine")
    results["held_reloaded_after_stop_s"] = wait(lambda: app.state()["held"] == "ready", 900, 1.0)
    say(f"held again {results['held_reloaded_after_stop_s']} s after the stop")
    results["with_held_2"] = one_run(app, "with held engine, second stream")
    c.prewarm = lambda key: None  # the old way: nothing loaded ahead
    c.held.drop(wait=True)
    results["without_held"] = one_run(app, "without held engine (before this change)")
    results["minutes"] = round((time.monotonic() - t0) / 60, 1)
    (OUT / "held.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "held_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
