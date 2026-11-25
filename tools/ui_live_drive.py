"""A real stream driven only through the StreamSafe window (RunPod job tools/ui_live_job.sh), in two parts:

1. The stream shows the StreamSafe window itself (a virtual screen, captured the way OBS would): every view is
   opened and every button pressed slowly, each press marked with a ring and a caption saying what it is.
2. Then the GTA 6 trailer, played ONCE, with the default settings (balanced, men not covered), nothing touched.
Then Stop, timed, and the report the window shows.

Env: STREAMSAFE_DATA_DIR, STREAM_URL (whole address with the key), TRAILER, OUT, DISPLAY (Xvfb), STEP_S.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from streamsafe import sensitivity
from streamsafe.app import engine as eng_mod
from streamsafe.app import runtime_select
from streamsafe.app.webui import server

OUT = Path(os.environ.get("OUT", "/workspace/res"))
W, H = 1600, 900
STEP_S = float(os.environ.get("STEP_S", "6"))
T0 = [0.0]

MARK_JS = """
(() => {
  const s = document.createElement('style');
  s.textContent = `.e2e-ring{position:fixed;width:46px;height:46px;margin:-23px 0 0 -23px;border:4px solid #ffd84d;border-radius:50%;
    pointer-events:none;z-index:9999;animation:e2e .9s ease-out forwards}
    @keyframes e2e{from{transform:scale(.4);opacity:1}to{transform:scale(1.6);opacity:0}}
    #e2e-cap{position:fixed;left:50%;top:8px;transform:translateX(-50%);z-index:9999;background:#ffd84d;color:#111;
    font:600 18px 'Readex Pro',sans-serif;padding:6px 16px;border-radius:10px;pointer-events:none;direction:rtl}`;
  document.head.append(s);
  const cap = document.createElement('div'); cap.id = 'e2e-cap'; document.body.append(cap);
  document.addEventListener('pointerdown', (e) => { const r = document.createElement('div'); r.className = 'e2e-ring';
    r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px'; document.body.append(r);
    setTimeout(() => r.remove(), 1000); }, true);
  window.e2eCaption = (t) => { cap.textContent = t; };
})();
"""


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


def pod_runtime() -> None:
    """The pod's own CUDA / TensorRT from pip (the app's NVIDIA pack is the Windows one)."""
    real_env = eng_mod.engine_env

    def pod_env(*args, **kw):
        env = real_env(*args, **kw)
        env.pop(runtime_select.RUNTIME_ENV, None)
        env.update({"TRT": "1", "TRT_BUILD": "0"})
        return env

    eng_mod.engine_env = pod_env


def ffmpeg_obs(args: list[str], port: int) -> subprocess.Popen:
    """OBS stand-in: whatever ``args`` read, sent to the app's SRT listener like OBS does."""
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *args, "-c:v", "libx264", "-preset", "veryfast", "-b:v", "6000k",
         "-g", "60", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-f", "mpegts", f"srt://127.0.0.1:{port}?mode=caller"],
    )  # fmt: skip


def drag_box(page: Page) -> None:
    b = page.locator(".pic-shot").bounding_box()
    page.mouse.move(b["x"] + b["width"] * 0.05, b["y"] + b["height"] * 0.05)
    page.mouse.down()
    page.mouse.move(b["x"] + b["width"] * 0.35, b["y"] + b["height"] * 0.3, steps=12)
    page.mouse.up()


def remove_area(page: Page, kind: str) -> None:
    page.click(f".area-btn[data-kind={kind}]")
    page.wait_for_timeout(1500)
    page.locator(f".zone.{kind}.active .x").first.click()
    page.wait_for_timeout(800)
    page.click("#zone-done")


def tour(sens) -> list:
    return [
        ("الحساسية: خفيفة", lambda pg: sens.nth(0).click()),
        ("الحساسية: عالية", lambda pg: sens.nth(2).click()),
        ("الحساسية: متوازنة (المقترحة)", lambda pg: sens.nth(1).click()),
        ("«لبّس الكل»: تشغيل", lambda pg: pg.locator("#sw-everyone").click(force=True)),
        ("«لبّس الكل»: إطفاء", lambda pg: pg.locator("#sw-everyone").click(force=True)),
        ("«غطِّ الشاشة كاملة»: تشغيل", lambda pg: pg.locator("#sw-full").click(force=True)),
        ("«غطِّ الشاشة كاملة»: إطفاء", lambda pg: pg.locator("#sw-full").click(force=True)),
        ("زر الطوارئ: البث كله يتغطى", lambda pg: pg.click("#panic-btn")),
        ("زر الطوارئ مرة ثانية: يرجع البث", lambda pg: pg.click("#panic-btn")),
        ("تحديد مكان الكاميرا", lambda pg: pg.click(".area-btn[data-kind=cam]")),
        ("«تم»: انحفظ مكان الكاميرا", lambda pg: pg.click("#zone-done")),
        ("تغطية جزء دائمًا", lambda pg: pg.click(".area-btn[data-kind=cover]")),
        ("«تم»: الجزء صار مغطى في البث", lambda pg: pg.click("#zone-done")),
        ("حذف الجزء المغطى (×)", lambda pg: remove_area(pg, "cover")),
        ("حذف مكان الكاميرا (×)", lambda pg: remove_area(pg, "cam")),
        ("فتح «تغطية دائمة»", lambda pg: pg.click("[data-view=pictures]")),
        ("«أضف صورة»: تصوير الشاشة", lambda pg: pg.click("#pic-new")),
        ("تحديد الصورة بالسحب", drag_box),
        ("كتابة اللعبة والاسم", lambda pg: (pg.fill("#pic-game", "GTA 6"), pg.fill("#pic-name", "تجربة"))),
        ("«احفظ»: الصورة انضافت", lambda pg: pg.click("#pic-save")),
        ("طفّ الصورة", lambda pg: pg.locator(".pic input[type=checkbox]").first.click(force=True)),
        ("حذف الصورة", lambda pg: pg.locator(".pic .del").first.click()),
        ("فتح «الإعدادات»", lambda pg: pg.click("[data-view=settings]")),
        ("منصة ثانية: كيك", lambda pg: pg.locator("#set-plat2 .plat").nth(1).click()),
        ("منصة ثانية: بدون", lambda pg: pg.locator("#set-plat2 .plat").first.click()),
        ("زر الطوارئ من الكيبورد: Ctrl+F9", lambda pg: pg.select_option("#panic-key", "Ctrl+F9")),
        ("زر الطوارئ: F9", lambda pg: pg.select_option("#panic-key", "F9")),
        ("«تذكّر المفتاح»: تغيير", lambda pg: pg.locator("#sw-remember").click(force=True)),
        ("«تذكّر المفتاح»: رجوع", lambda pg: pg.locator("#sw-remember").click(force=True)),
        ("تفعيل برو بمفتاح غلط: يطلع السبب", lambda pg: (pg.fill("#pass", "SSP-wrong.key"), pg.click("#pass-btn"))),
        ("فتح «الفحص»", lambda pg: pg.click("[data-view=check]")),
        ("«افحص مرة ثانية»", lambda pg: pg.click("#recheck")),
        ("فتح «الإعداد»", lambda pg: pg.click("[data-view=setup]")),
        ("«التالي»", lambda pg: pg.click("#step-next")),
        ("«رجوع»", lambda pg: pg.click("#step-back")),
        ("اللغة: English", lambda pg: pg.click("#lang")),
        ("اللغة: عربي", lambda pg: pg.click("#lang")),
        ("رجوع لشاشة البث", lambda pg: pg.click("[data-view=live]")),
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pod_runtime()
    app = server.WebApp()
    c = app.ctrl
    c.check_updates()  # the decision files (heads) through the app's own signed update channel
    if c.update_wanted()[0]:
        try:
            c.apply_update(lambda *a: None, threading.Event())  # data only: this copy cannot run the Windows installer
            say("heads updated through the update channel")
        except Exception as exc:
            say(f"heads update failed: {type(exc).__name__}: {exc}")
    app.do_platform({"platform": "kick", "key": os.environ["STREAM_URL"].strip()})
    s = c.settings
    s.follow_obs, s.sens_scale, s.dress_everyone, s.cover_mode, s.zones, s.covers = False, sensitivity.DEFAULT, False, "people", [], []
    c.save_settings()
    c.run_check()
    say(f"ready: platform={s.platform} key={'yes' if app.key else 'NO'} models={c.models_problem() or 'ok'}")

    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app, "live", [time.monotonic()]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t=live"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--kiosk", f"--window-size={W},{H}", "--window-position=0,0"])
        ctx = browser.new_context(no_viewport=True, record_video_dir=str(OUT / "video"), record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        page.on("pageerror", lambda e: say(f"PAGE ERROR {e}"))
        page.goto(url)
        page.wait_for_timeout(2500)
        page.evaluate(MARK_JS)
        page.click("[data-view=live]")

        def shot(name: str) -> None:
            page.screenshot(path=str(OUT / f"{name}.png"))

        def caption(text: str) -> None:
            page.evaluate("t => window.e2eCaption(t)", text)

        def step(text: str, do, settle: float = STEP_S) -> None:
            say(text)
            caption(text)
            page.wait_for_timeout(1200)  # the caption first, then the press
            try:
                do(page)
            except Exception as exc:
                say(f"  FAILED: {type(exc).__name__}: {exc}")
                caption("❌ " + text)
            page.wait_for_timeout(int(settle * 1000))

        # --- part 1: the window itself on the stream -----------------------------------------------------------
        step("ضغط «ابدأ الحماية والبث»", lambda pg: pg.click("#main-btn"), 1)
        if not wait(lambda: c.engine.running, 30):
            say("ENGINE DID NOT START: " + app.message[0])
            shot("start_failed")
            return 1
        port = c.settings.srt_port
        screen = ["-f", "x11grab", "-framerate", "30", "-video_size", f"{W}x{H}", "-i", os.environ["DISPLAY"],
                  "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]  # fmt: skip
        obs = ffmpeg_obs(screen, port)
        live = wait(lambda: c.link.platform == "connected", 120)
        T0[0] = time.monotonic()
        say(("LIVE ON KICK" if live else "NOT CONNECTED") + f" ({time.strftime('%H:%M:%S')} UTC): part 1, the window")
        caption("البث شغّال: الحين نجرّب كل زر")
        page.wait_for_timeout(8000)
        steps = tour(page.locator("#sens button")) if os.environ.get("TOUR", "1") == "1" else []
        for i, (what, do) in enumerate(steps):
            step(what, do)
            st = app.state()
            say(f"  live: engine={'on' if st['running'] else 'OFF'} platform={c.link.platform} panic={st['panic']}")
            shot(f"{i + 2:02d}")

        # --- part 2: the trailer once, default settings, nothing touched -------------------------------------
        s = c.settings
        defaults = (
            s.sens_scale == sensitivity.DEFAULT and not s.dress_everyone and s.cover_mode == "people" and not s.zones and not s.covers
        )
        say(f"before the trailer: sens={s.sens_scale} everyone={s.dress_everyone} mode={s.cover_mode} zones={s.zones} "
            f"covers={s.covers} pictures={len(c.pictures.items)} -> {'DEFAULTS' if defaults else 'NOT DEFAULT'}")  # fmt: skip
        caption("الحين التريلر مرة وحدة، بالإعداد الطبيعي، بدون لمس أي زر")
        page.wait_for_timeout(4000)
        obs.terminate()
        obs.wait(10)
        say("part 2: the trailer, once")
        obs = ffmpeg_obs(["-re", "-i", os.environ["TRAILER"]], port)
        n = 0
        while obs.poll() is None:  # what the viewers saw: the engine's own preview of the protected picture
            time.sleep(2)
            pf = c.preview_file()
            if pf.is_file():
                n += 1
                (OUT / "viewers").mkdir(exist_ok=True)
                (OUT / "viewers" / f"v{n:03d}.png").write_bytes(pf.read_bytes())
        say("trailer finished")
        page.wait_for_timeout(5000)

        # --- stop ------------------------------------------------------------------------------------------------
        t_stop = time.monotonic()
        page.click("#main-btn")
        wait(lambda: not c.engine.running, 60, 0.2)
        say(f"engine ended {time.monotonic() - t_stop:.1f} s after Stop (exit code {c.engine.exit_code})")
        report = wait(lambda: page.locator("#report").is_visible(), 60)
        say("report shown under the picture" if report else "REPORT NOT SHOWN")
        shot("99_report")
        (OUT / "engine_tail.txt").write_text("\n".join(c._engine_lines), encoding="utf-8")
        st = app.state()
        if st.get("report"):
            (OUT / "report_in_window.txt").write_text(st["report"]["text"], encoding="utf-8")
        ctx.close()
        browser.close()
    (OUT / "final_state.json").write_text(json.dumps(app.state(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    say("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
