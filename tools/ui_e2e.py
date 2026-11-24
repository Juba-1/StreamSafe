"""Every control of the StreamSafe window, pressed like a streamer would, checked twice: on the screen (what the
page shows) and in the app (what the server holds). Exit code 0 only when every check passed.

    python tools/ui_e2e.py --self                      # starts its own app server (development, Linux sandbox)
    python tools/ui_e2e.py --lock <data>/window.json   # the installed app's open window (Windows CI, real PC)

Options: --browser PATH (a Chromium/Edge exe) or --channel msedge; --out DIR (screenshots + results.json);
--live (also start "Try without streaming" and stop it: needs the models).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

RESULTS: list[tuple[bool, str]] = []


def check(ok: object, what: str) -> bool:
    RESULTS.append((bool(ok), what))
    print(("PASS  " if ok else "FAIL  ") + what, flush=True)
    return bool(ok)


class App:
    """The app's own answers (/api/state), to compare with the screen."""

    def __init__(self, url: str) -> None:
        self.base, _, q = url.partition("/?")
        self.token = q.split("t=", 1)[1].split("&")[0]

    def state(self) -> dict:
        req = urllib.request.Request(self.base + "/api/state", headers={"X-Token": self.token})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())


def wait_for(fn, timeout: float = 6.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if fn():
                return True
        except Exception:
            pass
        time.sleep(0.15)
    return False


def checked(page: Page, selector: str) -> list[str]:
    return [b.inner_text().replace("\n", " ") for b in page.locator(selector).all() if b.get_attribute("aria-checked") == "true"]


def run(page: Page, app: App, out: Path, live: bool) -> None:
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
    shot = lambda name: page.screenshot(path=str(out / f"{name}.png"))  # noqa: E731
    page.wait_for_timeout(1500)
    s = app.state()
    check("status" in s and "settings" in s, "the app answers /api/state")

    # --- every view opens -----------------------------------------------------------------------------------
    for view in ("live", "setup", "pictures", "settings", "check"):
        page.click(f"[data-view={view}]")
        check(wait_for(lambda v=view: page.locator(f"#view-{v}").is_visible()), f"menu: «{view}» opens")
        shot(f"view_{view}")

    # --- settings: platforms ---------------------------------------------------------------------------------
    page.click("[data-view=settings]")
    for pid in ("kick", "twitch", "youtube", "custom"):
        name = next(p["name"] for p in s["platforms"] if p["id"] == pid)
        page.locator("#set-plat1 .plat", has_text=name).click()
        ok_app = wait_for(lambda p=pid: app.state()["settings"]["platform"] == p)
        ok_screen = wait_for(lambda n=name: checked(page, "#set-plat1 .plat") == [n.replace("\n", " ")])
        check(ok_app and ok_screen, f"main platform «{pid}»: saved in the app and shown selected")
    check(page.locator("#set-plat1 input[placeholder^='rtmp']").count() == 1, "custom platform shows the server field")
    kick_name = next(p["name"] for p in s["platforms"] if p["id"] == "kick")
    page.locator("#set-plat1 .plat", has_text=kick_name).click()
    wait_for(lambda: app.state()["settings"]["platform"] == "kick")
    wait_for(lambda: checked(page, "#set-plat1 .plat") == [kick_name])
    page.wait_for_timeout(500)
    page.locator("#set-plat1 input[type=password]").fill("sk_us-west-2_e2eTESTKEYnotreal")
    page.locator("#set-plat1 button.secondary").click()
    check(wait_for(lambda: app.state()["settings"]["has_key"]), "stream key: Save keeps it in the app")
    check(wait_for(lambda: page.locator(".toast").count() > 0), "stream key: Save shows a confirmation")
    page.locator("#set-plat1 button.ghost").click()  # Test connection: an answer must appear (refused key = fine)
    check(wait_for(lambda: page.locator("#set-plat1 .result").inner_text().strip() != "", 25), "Test connection shows an answer")
    shot("settings_platform")

    kick = next(p["name"] for p in s["platforms"] if p["id"] == "kick")
    page.locator("#set-plat2 .plat", has_text=kick).click()
    check(
        wait_for(lambda: app.state()["settings"]["platform2"] == "kick") and wait_for(lambda: checked(page, "#set-plat2 .plat") == [kick]),
        "second platform «kick»: saved and shown",
    )
    page.locator("#set-plat2 .plat").first.click()  # none
    check(wait_for(lambda: app.state()["settings"]["platform2"] == ""), "second platform «none»: saved")

    # --- settings: panic key, switches, pass ----------------------------------------------------------------
    page.select_option("#panic-key", "Ctrl+F9")
    check(wait_for(lambda: app.state()["settings"]["panic_button"] == "Ctrl+F9"), "panic key Ctrl+F9 saved")
    page.select_option("#panic-key", "F9")
    for sw, key in (("#sw-follow", "follow_obs"), ("#sw-remember", "remember")):
        before = app.state()["settings"][key]
        page.locator(sw).click(force=True)
        check(wait_for(lambda k=key, b=before: app.state()["settings"][k] != b), f"switch «{key}» changes in the app")
        check(wait_for(lambda s=sw, b=before: page.locator(s).is_checked() != b), f"switch «{key}» changes on screen")
        page.locator(sw).click(force=True)
        wait_for(lambda k=key, b=before: app.state()["settings"][k] == b)
    page.fill("#pass", "SSP-not.a-real-key")
    page.click("#pass-btn")
    check(wait_for(lambda: page.locator("#plan-result").is_visible() and page.locator("#plan-result").inner_text().strip(), 15),
          "Activate with a wrong key shows why, in the card")  # fmt: skip
    check(page.locator("#plan-text").inner_text().strip() != "", "subscription line is filled")
    shot("settings_after")

    # --- live: sensitivity, switches, panic, screen areas ---------------------------------------------------
    page.click("[data-view=live]")
    for i, k in enumerate((1.7, 1.3, 1.0)):
        page.locator("#sens button").nth(i).click()
        check(wait_for(lambda k=k: abs(app.state()["settings"]["sens"] - k) < 1e-6), f"sensitivity {k} saved")
        check(wait_for(lambda i=i: page.locator("#sens button").nth(i).get_attribute("aria-checked") == "true"), f"sensitivity {k} shown")
    page.locator("#sens button").nth(1).click()
    for sw, key, want in (("#sw-everyone", "everyone", True), ("#sw-full", "cover_mode", "screen")):
        page.locator(sw).click(force=True)
        check(wait_for(lambda k=key, v=want: app.state()["settings"][k] == v), f"live switch «{key}» on")
        page.locator(sw).click(force=True)
        wait_for(lambda k=key, v=want: app.state()["settings"][k] != v)
    page.click("#panic-btn")
    check(
        wait_for(lambda: app.state()["panic"]) and wait_for(lambda: page.locator("#panic-veil").is_visible()),
        "panic button covers (app + screen)",
    )
    shot("live_panic")
    page.click("#panic-btn")
    check(
        wait_for(lambda: not app.state()["panic"]) and wait_for(lambda: page.locator("#panic-veil").is_hidden()),
        "panic button again uncovers",
    )
    for kind, key in (("cam", "zones"), ("cover", "covers")):
        n = len(app.state()["settings"][key])
        page.click(f".area-btn[data-kind={kind}]")
        page.wait_for_timeout(300)
        shot(f"live_area_{kind}")
        page.click("#zone-done")
        check(wait_for(lambda k=key, n=n: len(app.state()["settings"][k]) == max(n, 1)), f"screen area «{kind}»: marking saves it")
        page.click(f".area-btn[data-kind={kind}]")
        page.locator(f".zone.{kind}.active .x").first.click()
        page.click("#zone-done")
        check(wait_for(lambda k=key, n=n: len(app.state()["settings"][k]) == max(n - 1, 0) if n else not app.state()["settings"][k]),
              f"screen area «{kind}»: × removes it")  # fmt: skip

    # --- pictures, check, setup, language --------------------------------------------------------------------
    page.click("[data-view=pictures]")
    page.click("#pic-new")
    if not app.state()["running"]:
        check(wait_for(lambda: page.locator("#pic-need").is_visible()), "add picture without protection running explains what to do")
    page.click("[data-view=check]")
    page.click("#recheck")
    check(wait_for(lambda: page.locator("#checks li").count() > 0, 30), "system check lists its results")
    shot("check")
    page.click("[data-view=setup]")
    page.click("#step-next")
    check(wait_for(lambda: page.locator("#steps li.now").inner_text().strip().startswith("2")), "setup: Next goes to step 2")
    page.click("#step-back")
    check(wait_for(lambda: page.locator("#steps li.now").inner_text().strip().startswith("1")), "setup: Back goes to step 1")
    for _ in range(2):
        page.click("#step-next")
    page.wait_for_timeout(400)
    st = app.state()["setup"]
    if not st["obs_found"]:
        check(wait_for(lambda: page.locator("#step-body .note").count() > 0), "setup OBS step: says OBS is not installed")
    elif not st["obs_set"]:
        page.locator("#step-body button.primary").click()
        check(wait_for(lambda: app.state()["setup"]["obs_set"], 15), "setup OBS step: the button sets OBS up")
    shot("setup_obs")
    first = app.state()["lang"]
    is_dir = lambda lg: page.evaluate("document.documentElement.dir") == ("ltr" if lg == "en" else "rtl")  # noqa: E731
    other = "en" if first == "ar" else "ar"
    page.click("#lang")
    check(wait_for(lambda: app.state()["lang"] == other and is_dir(other)),
          f"language button: {first} to {other} (app + screen)")  # fmt: skip
    shot("language_" + other)
    page.click("#lang")
    check(wait_for(lambda: app.state()["lang"] == first and is_dir(first)),
          f"language button: back to {first}")  # fmt: skip

    # --- protection on and off (needs the models) ------------------------------------------------------------
    page.click("[data-view=live]")
    if live:
        page.click("#trial-btn")
        check(wait_for(lambda: app.state()["running"], 60), "Try without streaming: protection starts")
        check(wait_for(lambda: page.locator("#preview").is_visible(), 60), "the viewers' picture shows in the window")
        shot("live_trial")
        for view in ("settings", "pictures", "check", "live"):  # switching views while protected
            page.click(f"[data-view={view}]")
            page.wait_for_timeout(800)
            check(app.state()["running"], f"still protecting after opening «{view}»")
        page.click("#main-btn")
        check(wait_for(lambda: not app.state()["running"], 40), "Stop ends protection")
        check(wait_for(lambda: page.locator("#report").is_visible(), 30), "the session report shows under the picture")
        shot("live_report")
    else:
        disabled = page.locator("#trial-btn").is_disabled()
        check(disabled == bool(app.state()["setup"]["models"]), "Try button is off only while the models are missing")
    check(not errors, "no errors in the page" + ("" if not errors else ": " + " | ".join(errors[:5])))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles are not UTF-8 by default
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--self", action="store_true")
    ap.add_argument("--lock", default="")
    ap.add_argument("--browser", default="")
    ap.add_argument("--channel", default="")
    ap.add_argument("--out", default="ui_e2e")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.self:
        from streamsafe.app.webui import server

        web = server.WebApp()
        srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(web, "e2e", [time.monotonic()]))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/?t=e2e"
    else:
        lock = Path(args.lock)
        wait_for(lock.is_file, 60)
        url = json.loads(lock.read_text(encoding="utf-8"))["url"]
    with sync_playwright() as p:
        kw = {"executable_path": args.browser} if args.browser else {"channel": args.channel} if args.channel else {}
        browser = p.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": 1280, "height": 820})
        page.goto(url)
        try:
            run(page, App(url), out, args.live)
        except Exception as exc:
            check(False, f"run stopped: {type(exc).__name__}: {exc}")
            page.screenshot(path=str(out / "stopped.png"))
        browser.close()
    failed = [w for ok, w in RESULTS if not ok]
    (out / "results.json").write_text(
        json.dumps({"passed": len(RESULTS) - len(failed), "failed": failed}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
