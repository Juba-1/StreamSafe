"""A real stream driven only through the StreamSafe window (RunPod job tools/ui_live_job.sh).

The app is set up the way a streamer does it (models and heads through the app's own download and update, the
platform by pasting the whole address), then a browser presses the window's buttons: Start, every view, every
live control (sensitivity, dress everyone, whole screen, panic, screen areas), Stop. OBS is played by FFmpeg
sending the GTA 6 trailer over SRT. Every step is printed with the time since the stream went live, so the
owner can watch the channel and match what he sees. The browser records a video of the window.

Env: STREAMSAFE_DATA_DIR, STREAM_URL (the platform's whole address with the key), TRAILER, OUT, TOUR_S.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from streamsafe.app.webui import server

OUT = Path(os.environ.get("OUT", "/workspace/res"))
T0 = [0.0]


def say(text: str) -> None:
    t = time.monotonic() - T0[0] if T0[0] else 0.0
    print(f"STEP t={t:6.1f}s  {text}", flush=True)


def wait(fn, timeout: float, every: float = 0.5) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if fn():
                return True
        except Exception:
            pass
        time.sleep(every)
    return False


def job_done(app: server.WebApp, timeout: float) -> str:
    wait(lambda: not app.job_name, timeout, 2)
    return app.message[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = server.WebApp()
    c = app.ctrl
    # --- set up like a streamer: models, heads (update channel), platform --------------------------------------
    if c.models_problem():
        say("downloading the models through the app")
        app.do_models({})
        say("models: " + job_done(app, 3600))
    c.check_updates()  # the decision files (heads) come through the app's own signed update channel
    if c.update_wanted()[0]:
        try:
            c.apply_update(lambda *a: None, threading.Event())  # data only: this dev copy cannot run the Windows installer
            say("heads updated through the update channel")
        except Exception as exc:  # noqa: BLE001
            say(f"heads update failed: {type(exc).__name__}: {exc}")
    app.do_platform({"platform": "kick", "key": os.environ["STREAM_URL"].strip()})
    c.settings.follow_obs = False  # FFmpeg plays OBS here: no OBS settings files on the pod
    c.save_settings()
    c.run_check()
    c.use_nvidia = True  # the pod's own CUDA / TensorRT (pip), not the Windows NVIDIA pack
    say(f"ready: platform={c.settings.platform} key={'yes' if app.key else 'NO'} models={c.models_problem() or 'ok'}")

    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app, "live", [time.monotonic()]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t=live"
    obs = None
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 820}, record_video_dir=str(OUT / "video"),
                                  record_video_size={"width": 1280, "height": 820})  # fmt: skip
        page = ctx.new_page()
        page.on("pageerror", lambda e: say(f"PAGE ERROR {e}"))
        page.goto(url)
        page.wait_for_timeout(2000)
        page.click("[data-view=live]")
        shot = lambda n: page.screenshot(path=str(OUT / f"{n}.png"))  # noqa: E731
        shot("00_before")

        say("press Start (protection + stream)")
        page.click("#main-btn")
        if not wait(lambda: c.engine.running, 30):
            say("ENGINE DID NOT START: " + app.message[0])
            shot("start_failed")
            return 1
        port = c.settings.srt_port
        obs = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", os.environ["TRAILER"],
             "-c:v", "libx264", "-preset", "veryfast", "-b:v", "6000k", "-g", "60", "-c:a", "aac", "-b:a", "160k",
             "-f", "mpegts", f"srt://127.0.0.1:{port}?mode=caller"],
        )  # fmt: skip
        live = wait(lambda: c.link.platform == "connected", 120)
        T0[0] = time.monotonic()
        say(("LIVE ON KICK" if live else "NOT CONNECTED: " + str(app.state().get("link"))) + f" ({time.strftime('%H:%M:%S')} UTC)")
        shot("01_live")
        tour = [
            ("sensitivity: light", lambda: page.locator("#sens button").nth(0).click()),
            ("sensitivity: high", lambda: page.locator("#sens button").nth(2).click()),
            ("sensitivity: balanced", lambda: page.locator("#sens button").nth(1).click()),
            ("dress everyone ON (men covered too)", lambda: page.locator("#sw-everyone").click(force=True)),
            ("dress everyone OFF", lambda: page.locator("#sw-everyone").click(force=True)),
            ("whole screen mode ON", lambda: page.locator("#sw-full").click(force=True)),
            ("whole screen mode OFF", lambda: page.locator("#sw-full").click(force=True)),
            ("PANIC ON: the whole stream must be covered now", lambda: page.click("#panic-btn")),
            ("PANIC OFF: picture back", lambda: page.click("#panic-btn")),
            ("always-cover area: bottom-left box ON", lambda: (page.click(".area-btn[data-kind=cover]"), page.click("#zone-done"))),
            (
                "always-cover area OFF",
                lambda: (
                    page.click(".area-btn[data-kind=cover]"),
                    page.locator(".zone.cover.active .x").first.click(),
                    page.click("#zone-done"),
                ),
            ),
            ("webcam area marked (bottom-right)", lambda: (page.click(".area-btn[data-kind=cam]"), page.click("#zone-done"))),
            ("open Settings while live", lambda: page.click("[data-view=settings]")),
            ("open Always cover while live", lambda: page.click("[data-view=pictures]")),
            ("open Check while live", lambda: page.click("[data-view=check]")),
            ("open Setup while live", lambda: page.click("[data-view=setup]")),
            ("language English", lambda: page.click("#lang")),
            ("language Arabic", lambda: page.click("#lang")),
            ("back to Live", lambda: page.click("[data-view=live]")),
        ]
        gap = float(os.environ.get("TOUR_S", "240")) / len(tour)
        for i, (what, do) in enumerate(tour):
            say(what)
            try:
                do()
            except Exception as exc:
                say(f"  FAILED: {type(exc).__name__}: {exc}")
            page.wait_for_timeout(int(gap * 1000))
            st = app.state()
            say(f"  still live: engine={'on' if st['running'] else 'OFF'} platform={c.link.platform} panic={st['panic']}")
            shot(f"{i + 2:02d}_{what.split(':')[0].replace(' ', '_')[:24]}")
        say("press Stop")
        page.click("#main-btn")
        wait(lambda: not c.engine.running, 60)
        report = wait(lambda: page.locator("#report").is_visible(), 60)
        say("report shown under the picture" if report else "REPORT NOT SHOWN")
        shot("99_report")
        if app.state().get("report"):
            (OUT / "report_in_window.txt").write_text(app.state()["report"]["text"], encoding="utf-8")
        ctx.close()
        browser.close()
    if obs:
        obs.kill()
    (OUT / "final_state.json").write_text(json.dumps(app.state(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    say("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
