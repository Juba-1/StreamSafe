"""The StreamSafe window as a local web page (the new interface, 3.0.0.40+).

The page (static/) runs in Microsoft Edge's app mode (a plain window, no address bar; Edge ships with Windows
10/11) and talks to this process over 127.0.0.1 with a random port and a random token. All the logic stays in
``controller.Controller``; this module only turns its state into JSON and the page's clicks into calls.

The engine is a separate process: nothing here can slow the stream. The page polls ``/api/state`` (cheap) and
``/api/preview`` (the small picture the engine writes every 2 s).
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from streamsafe import sensitivity
from streamsafe.app import controller, engine, hotkey, i18n, ingest, modelpack, obs_setup, passkey, plan, preflight, release, updates
from streamsafe.app import settings as cfg

STATIC = Path(__file__).with_name("static")
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml", ".woff2": "font/woff2"}
STATUS = {  # engine status -> (i18n label, i18n hint, look)
    engine.STOPPED: ("st_stopped", "st_hint_stopped", "idle"),
    engine.STARTING: ("st_starting", "st_hint_covered", "warn"),
    engine.PROTECTED: ("st_protected", "st_hint_protected", "ok"),
    engine.COVERED: ("st_covered", "st_hint_covered", "warn"),
    engine.STALLED: ("st_stalled", "st_hint_covered", "bad"),
    engine.FAILED: ("st_failed", "st_hint_covered", "bad"),
}


def log_line(folder: Path, text: str) -> None:
    """data folder/webui.log: actions that failed and their traceback (kept small)."""
    with contextlib.suppress(OSError):
        f = folder / "webui.log"
        if f.is_file() and f.stat().st_size > 512_000:
            f.replace(folder / "webui.old.log")
        with f.open("a", encoding="utf-8") as out:
            out.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text.rstrip() + "\n")


class WebApp:
    """One window's worth of state on top of the controller (jobs, messages, the panic key)."""

    def __init__(self, ctrl: controller.Controller | None = None) -> None:
        self.ctrl = ctrl or controller.Controller()
        self.lang = self.ctrl.settings.lang if self.ctrl.settings.lang in i18n.LANGS else "ar"
        self.key = self.ctrl.stored_key()
        self.ctrl.key2 = self.ctrl.stored_key2()
        self.message = ("", "")  # (text, kind: ok / bad / info)
        self.message_id = 0  # the page shows each message once, in every view
        self.stopping = False
        self.started_at = time.time()
        self.job_name = ""
        self.progress = ("", -1.0)
        self.cancel = threading.Event()
        self.test_result = ("", "")
        self.lock = threading.Lock()
        self._hotkey: hotkey.Hotkey | None = None
        self._announced = False
        self.arm_panic_key()
        threading.Thread(target=self._first_checks, name="first-checks", daemon=True).start()

    # --- text ------------------------------------------------------------------------------------------------
    def tr(self, key: str, **values: object) -> str:
        return i18n.t(self.lang, key, **values) if key in i18n.TEXT else key

    def say(self, text: str, kind: str = "info") -> None:
        self.message = (text, kind)
        self.message_id += 1

    def log(self, text: str) -> None:
        log_line(self.ctrl.preview_file().parent, text)

    def safe(self, what: str, fn: Callable[[], object], default: object) -> object:
        """One part of the state; a failure there is logged and leaves the rest of the window working."""
        try:
            return fn()
        except Exception:
            self.log(f"state {what} failed:\n{traceback.format_exc()}")
            return default

    # --- background work -------------------------------------------------------------------------------------
    def _first_checks(self) -> None:
        with contextlib.suppress(Exception):
            passkey.sync_clock(self.ctrl.env)  # a supporter's key follows the internet's date, not only the PC clock
        with contextlib.suppress(Exception):
            self.ctrl.run_check()
        with contextlib.suppress(Exception):
            self.ctrl.check_updates()

    def job(self, name: str, work: Callable[[], str], ok_text: str | Callable[[], str] = "") -> dict:
        with self.lock:
            if self.job_name:
                return {"ok": False, "why": self.tr("test_busy")}
            self.job_name = name
            self.cancel.clear()
            self.progress = (self.tr("installing"), -1.0)

        def run() -> None:
            try:
                err = work()
            except Exception as exc:  # a worker must never die silently
                err = f"{type(exc).__name__}: {exc}"
            if err:
                self.log(f"job {name}: {err}")
            self.say(err or (ok_text() if callable(ok_text) else ok_text), "bad" if err else "ok")
            with self.lock:
                self.job_name = ""
            with contextlib.suppress(Exception):
                self.ctrl.run_check()

        threading.Thread(target=run, name=name, daemon=True).start()
        return {"ok": True}

    def on_progress(self, what: str, done: int, total: int) -> None:
        word = self.tr("p_" + what) if "p_" + what in i18n.TEXT else what
        if total > 0:
            size = f" {done / 1e6:,.0f} / {total / 1e6:,.0f} MB" if total > 5_000_000 else f" {done}/{total}"
            self.progress = (word + size, min(1.0, done / total))
        else:
            self.progress = (word, -1.0)

    # --- the panic key ---------------------------------------------------------------------------------------
    def arm_panic_key(self) -> None:
        if self._hotkey is not None:
            self._hotkey.stop()
            self._hotkey = None
        name = self.ctrl.settings.panic_button
        if name:
            hk = hotkey.Hotkey(name, lambda: self.ctrl.set_panic(not self.ctrl.panic))
            if hk.start():
                self._hotkey = hk
            elif sys.platform == "win32":
                self.say(self.tr("panic_taken", hk=name), "bad")

    # --- state for the page ----------------------------------------------------------------------------------
    def state(self) -> dict:
        c, s, lk = self.ctrl, self.ctrl.settings, self.ctrl.link
        status = c.status()
        label, hint, look = STATUS[status]
        if status == engine.STOPPED and c.expired():
            label, hint, look = "st_blocked", "beta_expired", "bad"
        running = c.engine.running
        if status in (engine.STOPPED, engine.FAILED) and c.last_exit not in (None, 0) and not self._announced:
            self._announced = True
            self.say(self.tr(c.exit_reason(), code=c.last_exit), "bad")
        problem = c.models_problem()
        u = c.update
        data_new, app_new = c.update_wanted()
        pf = c.preview_file()
        r = c.check_result
        return {
            "lang": self.lang,
            "version": updates.current_version(),
            "status": {"key": status, "label": self.tr(label), "hint": self.tr(hint), "look": look},
            "running": running,
            "preview_mode": c.preview,
            "preview_at": pf.stat().st_mtime if running and pf.is_file() else 0,
            "link": self.safe("link", self.link_view, None) if running or lk.stopped else None,
            "message": {"text": self.message[0], "kind": self.message[1], "id": self.message_id},
            "stopping": self.stopping,
            "report": self.safe("report", self.report_view, None),
            "job": {"name": self.job_name, "text": self.progress[0], "pct": self.progress[1]} if self.job_name else None,
            "test": {"text": self.test_result[0], "kind": self.test_result[1]},
            "setup": {
                "models": ""
                if not problem
                else (self.tr("models_needed") if "not installed" in problem else self.tr("models_bad", why=problem)),
                "nvidia": bool(not problem and c.has_nvidia_gpu() and not c.use_nvidia and not s.nvidia_declined and not running),
                "nvidia_gb": release.NVIDIA_DOWNLOAD_GB,
                "obs_found": obs_setup.find_obs() is not None,
                "obs_set": obs_setup.is_set(c.obs_server()),
                "key_ok": not self.target_problems(),
            },
            "settings": {
                "platform": s.platform,
                "rtmp_url": s.rtmp_url,
                "has_key": bool(self.key),
                "platform2": s.platform2,
                "rtmp_url2": s.rtmp_url2,
                "has_key2": bool(c.key2),
                "remember": s.remember_key,
                "sens": s.sens_scale,
                "sens_levels": sensitivity.LEVELS,
                "everyone": s.dress_everyone,
                "cover_mode": s.cover_mode,
                "panic_button": s.panic_button,
                "zones": s.zones,
                "covers": s.covers,
                "follow_obs": s.follow_obs,
                "translate": s.translate_ar,
            },
            "panic": c.panic,
            "boxes": self.safe("boxes", self.live_boxes, []) if running else [],
            "pictures": [{k: i[k] for k in ("id", "game", "name", "rect", "full", "on")} for i in c.pictures.items],
            "allow": [[*a[:4], round(a[4] - time.time())] for a in c.live_allow()],
            "platforms": [{"id": p, "name": self.tr("plat_" + p)} for p in ingest.PLATFORMS],
            "panic_keys": hotkey.CHOICES,
            "obs_server": c.obs_server(),
            "update": {"app": app_new, "data": data_new, "version": u.app_version if u else "", "one_click": bool(u and u.app_sha256)}
            if (data_new or app_new)
            else None,
            "check": {"verdict": r.verdict, "error": r.error, "items": self.safe("check", lambda: self.check_items(r), [])},
            "plan": self.safe("plan", self.plan_view, {"pro": False}),
            "brand": {"pro": plan.allows("custom_brand", c.env), "set": bool(s.brand_image)},
            "days_left": c.days_left(),
            "tab": os.environ.get("STREAMSAFE_TAB", ""),
        }

    def report_view(self) -> dict | None:
        """The newest session report: when, and its text (the window shows it after a stop)."""
        f = self.ctrl.last_report()
        if f is None:
            return None
        text = f.read_text(encoding="utf-8", errors="replace")
        return {"name": f.name, "at": f.stat().st_mtime, "new": f.stat().st_mtime >= self.started_at, "text": text[:6000]}

    def live_boxes(self) -> list:
        """Where the person covers are on the latest preview (fresh only)."""
        try:
            data = json.loads(Path(str(self.ctrl.preview_file()) + ".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return data.get("boxes", []) if time.time() - float(data.get("t", 0)) < 5 else []

    def target_problems(self) -> list[str]:
        return list(self.ctrl.target(self.key).problems) if self.ctrl.settings.output == cfg.OUTPUT_RTMP else []

    def link_view(self) -> dict:
        c, s, lk = self.ctrl, self.ctrl.settings, self.ctrl.link
        server = c.obs_server()
        why = self.tr("why_" + lk.reason) if lk.reason else ""
        obs = {
            "connected": (self.tr("lk_obs_ok"), "ok"),
            "disconnected": (self.tr("lk_obs_lost", server=server), "warn"),
            "failed": (self.tr("lk_obs_failed", why=why), "bad"),
        }.get(lk.input, (self.tr("lk_obs_wait", server=server), "warn"))
        out = []
        if lk.protection:
            word, look = ("lk_prot_loading", "warn") if lk.protection == "loading" else ("lk_prot_ok", "ok")
            out.append({"name": self.tr("lk_prot"), "text": self.tr(word), "look": look})
        if c.preview:
            out.append({"name": self.tr("preview"), "text": self.tr("lk_window"), "look": "idle"})
        elif s.output == cfg.OUTPUT_RTMP:
            text, look = {
                "connecting": (self.tr("lk_plat_connecting"), "warn"),
                "connected": (self.tr("lk_plat_ok"), "ok"),
                "lost": (self.tr("lk_plat_lost", why=why, s=f"{lk.retry_s:g}"), "warn"),
                "failed": (self.tr("lk_plat_failed", why=why), "bad"),
            }.get(lk.platform, (self.tr("lk_plat_wait"), "idle"))
            out.append({"name": self.tr("plat_" + s.platform) if s.platform else "", "text": text, "look": look})
            if s.platform2:
                look2 = {"connected": "ok", "failed": "bad"}.get(lk.platform2, "warn")
                word = {"connected": "lk_plat_ok", "failed": "lk_plat_failed"}.get(lk.platform2, "lk_plat_connecting")
                out.append({"name": self.tr("plat_" + s.platform2), "text": self.tr(word, why=""), "look": look2})
        return {"obs": {"text": obs[0], "look": obs[1]}, "out": out}

    def check_items(self, r: controller.CheckResult) -> list[dict]:
        out = []
        for item in r.checks:
            name = item.get("name", "")
            out.append(
                {
                    "status": item.get("status", ""),
                    "name": self.tr("chk_" + name) if "chk_" + name in i18n.TEXT else name,
                    "detail": item.get("detail", ""),
                    "fix": item.get("fix_ar" if self.lang == "ar" else "fix_en", "") if item.get("status") != "ok" else "",
                }
            )
        return out

    def plan_view(self) -> dict:
        """The same words as the classic window's subscription line (pass, licence, or free)."""
        from streamsafe.app import license as lic

        env = self.ctrl.env
        held, sp = lic.load(env), passkey.load(env)
        if held is None and sp is not None:  # a supporter's pass: Pro until its last day
            why = passkey.why_off(env)
            if not why:
                return {"pro": True, "text": self.tr("pass_on", date=sp.expires.isoformat(), who=sp.who or "-"), "look": "ok"}
            return {"pro": False, "text": self.pass_reason(why, sp.expires.isoformat()), "look": "warn"}
        if held is not None:
            st = self.ctrl.license_status
            text = self.tr("license_on", plan=self.tr("plan_" + held.plan), key=lic.masked(held.key))
            return {"pro": held.plan == "pro", "text": text + (self.tr("license_grace") if st.grace else ""), "look": "ok"}
        if release.LICENSE_REQUIRED:
            return {"pro": False, "text": self.tr("license_needed"), "look": "bad"}
        return {"pro": False, "text": self.tr("license_beta"), "look": "idle"}

    def pass_reason(self, why: str, date: str) -> str:
        return {
            "invalid": self.tr("pass_invalid"),
            "clock": self.tr("pass_clock"),
            "expired": self.tr("pass_ended", date=date) if date else self.tr("pass_expired"),
        }.get(why, self.tr("pass_invalid"))

    # --- actions from the page -------------------------------------------------------------------------------
    def act(self, name: str, a: dict) -> dict:
        fn = getattr(self, "do_" + name, None)
        if fn is None:
            return {"ok": False, "why": "unknown action"}
        try:
            out = fn(a) or {"ok": True}
        except Exception as exc:
            self.log(f"action {name} failed:\n{traceback.format_exc()}")
            why = f"{type(exc).__name__}: {exc}"
            self.say(why, "bad")
            return {"ok": False, "why": why}
        if not out.get("ok", True) and out.get("why"):
            self.say(str(out["why"]), "bad")
        return out

    def do_lang(self, a: dict) -> None:
        self.lang = "en" if self.lang == "ar" else "ar"
        self.ctrl.settings.lang = self.lang
        self.ctrl.save_settings()

    def do_texts(self, a: dict) -> dict:
        return {"ok": True, "texts": {k: v[0 if self.lang == "ar" else 1] for k, v in i18n.TEXT.items()}}

    def do_platform(self, a: dict) -> None:
        s = self.ctrl.settings
        s.platform = str(a.get("platform", s.platform))
        if "rtmp_url" in a:
            s.rtmp_url = str(a["rtmp_url"]).strip()
        if "key" in a:
            s.rtmp_url, self.key = self.ctrl.take_pasted(s.rtmp_url, str(a["key"]).strip())
        self.test_result = ("", "")
        self.ctrl.save_settings()
        if not self.ctrl.remember(self.key):
            s.remember_key = False

    def do_platform2(self, a: dict) -> None:
        s = self.ctrl.settings
        s.platform2 = str(a.get("platform2", ""))
        s.rtmp_url2 = str(a.get("rtmp_url2", s.rtmp_url2)).strip()
        if "key2" in a:
            self.ctrl.key2 = str(a["key2"]).strip()
        self.ctrl.save_settings()
        self.ctrl.remember2(self.ctrl.key2)

    def do_test(self, a: dict) -> dict:
        problems = self.target_problems()
        if problems:
            self.test_result = (self.tr(problems[0]), "bad")
            return {"ok": False}
        if self.ctrl.engine.running or self.job_name:
            self.test_result = (self.tr("test_busy"), "bad")
            return {"ok": False}
        self.test_result = (self.tr("test_running"), "info")

        def work() -> None:
            why = self.ctrl.test_connection(self.key)
            self.test_result = (self.tr("test_fail", why=self.tr("why_" + why)), "bad") if why else (self.tr("test_ok"), "ok")

        threading.Thread(target=work, name="link-test", daemon=True).start()
        return {"ok": True}

    def do_start(self, a: dict) -> dict:
        preview = bool(a.get("preview"))
        c = self.ctrl
        self._announced = False
        self.message = ("", "")
        warns = [self.tr(w) for w in preflight.warnings(c.env)]
        if c.settings.follow_obs and c.settings.input == cfg.INPUT_SRT:
            engine.follow_obs(c.settings, obs_setup.obs_video())
            c.settings.obs_kbps = obs_setup.obs_kbps()
        try:
            c.start(self.key, preview=preview)
        except (RuntimeError, OSError) as exc:
            why = self.tr("sac_blocked") if preflight.blocked_by_windows(exc) else self.tr(str(exc))
            self.say(self.tr("not_ready", why=why), "bad")
            return {"ok": False}
        if c.settings.input == cfg.INPUT_SRT:
            threading.Thread(target=self.bring_obs, args=(c.obs_server(),), daemon=True).start()
        self.say("\n".join([self.tr("preview_on"), *warns] if preview else warns), "info")
        return {"ok": True}

    def bring_obs(self, server: str) -> None:
        if obs_setup.obs_running():
            self.say(self.tr("obs_press_start") if obs_setup.is_set(server) else self.tr("obs_auto_running"), "info")
            return
        why = "" if obs_setup.is_set(server) else obs_setup.apply(server)
        exe = obs_setup.find_obs()
        if why or exe is None:
            self.say(self.obs_text(why or "no_obs"), "bad")
        elif obs_setup.launch(exe):
            self.say(self.tr("obs_opening"), "info")

    def obs_text(self, why: str) -> str:
        return self.tr(
            {"": "obs_auto_done", "running": "obs_auto_running", "no_obs": "obs_auto_no_obs"}.get(why, "obs_auto_error"), why=why
        )

    def do_stop(self, a: dict) -> None:
        if self.stopping or not self.ctrl.engine.running:
            return
        self.stopping = True
        self.say(self.tr("stopping") if "stopping" in i18n.TEXT else "…", "info")

        def run() -> None:
            try:
                self.ctrl.stop()
            except Exception:
                self.log(f"stop failed:\n{traceback.format_exc()}")
            finally:
                self.stopping = False
                self.say(self.tr("stopped_report") if "stopped_report" in i18n.TEXT else "", "ok")

        threading.Thread(target=run, name="stop", daemon=True).start()

    def do_panic(self, a: dict) -> None:
        self.ctrl.set_panic(bool(a.get("on", not self.ctrl.panic)))

    def do_sens(self, a: dict) -> None:
        self.ctrl.set_sensitivity(sensitivity.clamp(a.get("k", sensitivity.DEFAULT)))

    def do_everyone(self, a: dict) -> None:
        self.ctrl.set_everyone(bool(a.get("on")))

    def do_cover_mode(self, a: dict) -> None:
        mode = str(a.get("mode"))
        self.ctrl.settings.cover_mode = mode if mode in (cfg.COVER_PEOPLE, cfg.COVER_SCREEN) else cfg.COVER_PEOPLE
        self.ctrl.save_settings()

    def do_zones(self, a: dict) -> None:
        self.ctrl.set_zones(list(a.get("zones", [])))

    def do_release(self, a: dict) -> None:
        self.ctrl.release(list(a.get("box", [])), 60.0)

    def do_restore(self, a: dict) -> None:
        self.ctrl.restore(a.get("i") if isinstance(a.get("i"), int) else None)

    # --- pictures always covered (per game) ---
    def do_pic_freeze(self, a: dict) -> dict:
        if not self.ctrl.engine.running:
            return {"ok": False, "why": "not_running"}
        return {"ok": self.ctrl.pictures.freeze(self.ctrl.preview_file())}

    def do_pic_add(self, a: dict) -> dict:
        why = self.ctrl.pictures.add(str(a.get("game", "")), str(a.get("name", "")), list(a.get("rect", [])), bool(a.get("full")))
        self.ctrl.write_control()
        return {"ok": not why, "why": self.tr(why) if why else ""}

    def do_pic_del(self, a: dict) -> None:
        self.ctrl.pictures.remove(str(a.get("id", "")))
        self.ctrl.write_control()

    def do_pic_set(self, a: dict) -> None:
        values = {k: a[k] for k in ("on", "full") if k in a}
        if "game" in a:
            self.ctrl.pictures.set_game(str(a["game"]), bool(a.get("on")))
        else:
            self.ctrl.pictures.set(str(a.get("id", "")), **values)
        self.ctrl.write_control()

    def brand_upload(self, data: bytes) -> dict:
        if not plan.allows("custom_brand", self.ctrl.env):
            return {"ok": False, "why": "Pro"}
        if not data:
            return {"ok": False, "why": self.tr("brand_bad")}
        tmp = self.ctrl.preview_file().parent / "upload_logo.bin"
        tmp.write_bytes(data)
        why = self.ctrl.set_brand_image(str(tmp))
        with contextlib.suppress(OSError):
            tmp.unlink()
        return {"ok": not why, "why": self.tr(why) if why else ""}

    def do_brand_clear(self, a: dict) -> None:
        self.ctrl.set_brand_image("")

    def do_covers(self, a: dict) -> None:
        self.ctrl.set_covers(list(a.get("covers", [])))

    def do_panic_key(self, a: dict) -> None:
        name = str(a.get("key", ""))
        if name and hotkey.parse(name) is None:
            return
        self.ctrl.settings.panic_button = name
        self.ctrl.save_settings()
        self.arm_panic_key()

    def do_toggle(self, a: dict) -> None:
        name = str(a.get("name"))
        if name in ("follow_obs", "remember_key", "translate_ar"):
            setattr(self.ctrl.settings, name, bool(a.get("on")))
            self.ctrl.save_settings()

    def do_obs_setup(self, a: dict) -> None:
        why = obs_setup.apply(self.ctrl.obs_server())
        self.say(self.obs_text(why), "bad" if why else "ok")

    def do_obs_restore(self, a: dict) -> None:
        why = obs_setup.restore()
        key = {"": "obs_restore_done", "no_backup": "obs_restore_none", "running": "obs_restore_running"}.get(why, "obs_auto_error")
        self.say(self.tr(key), "ok" if not why else "bad")

    def do_models(self, a: dict) -> dict:
        def work() -> str:
            try:
                self.ctrl.install_models("download", "", self.on_progress, self.cancel)
            except modelpack.SourceError as exc:
                return self.tr(f"src_{exc.kind}", why=str(exc))
            except modelpack.PackError as exc:
                return self.tr("pack_bad", why=str(exc))
            except Exception as exc:
                return self.tr("install_cancelled") if self.cancel.is_set() else self.tr("src_other", why=str(exc))
            return ""

        return self.job("models", work, self.tr("install_done"))

    def do_nvidia(self, a: dict) -> dict:
        def work() -> str:
            why = self.ctrl.enable_nvidia(
                lambda w, d, t: self.on_progress("nvidia_" + ("downloading" if t > 1000 else "preparing"), d, t), self.cancel
            )
            return self.tr("nvidia_failed", why=why) if why else ""

        return self.job("nvidia", work, self.tr("nvidia_ready"))

    def do_nvidia_no(self, a: dict) -> None:
        self.ctrl.settings.nvidia_declined = True
        self.ctrl.save_settings()

    def do_cancel(self, a: dict) -> None:
        self.cancel.set()

    def do_update(self, a: dict) -> dict:
        if self.ctrl.engine.running:
            self.say(self.tr("update_busy"), "bad")
            return {"ok": False}
        _data_new, app_new = self.ctrl.update_wanted()
        if app_new and self.ctrl.update and self.ctrl.update.app_sha256:

            def work_app() -> str:
                path = self.ctrl.download_app_update(lambda d, t: self.on_progress("update", d, t), self.cancel)
                self.ctrl.run_installer(path)
                threading.Timer(1.0, lambda: os._exit(0)).start()  # the installer replaces and reopens us
                return ""

            return self.job("update", work_app, self.tr("update_done"))

        def work() -> str:
            try:
                self.ctrl.apply_update(lambda w, d, t: self.on_progress(w, d, t), self.cancel)
            except updates.UpdateError as exc:
                return self.tr("update_failed", why=str(exc))
            return ""

        return self.job("update", work, self.tr("update_done"))

    def do_recheck(self, a: dict) -> dict:
        return self.job("check", lambda: self.ctrl.run_check().error, "")

    def do_pass(self, a: dict) -> dict:
        key = str(a.get("key", "")).strip()

        def work() -> str:
            if not key:
                return self.tr("pass_invalid")
            if passkey.is_pass(key):  # a supporter's pass: checked here (and the internet's date)
                why = passkey.activate(key, self.ctrl.env)
                return self.pass_reason(why, "") if why else ""
            st = self.ctrl.activate_license(key)
            return "" if st.ok else self.tr("why_license_" + (st.reason or "invalid"))

        def done() -> str:
            return str(self.plan_view()["text"])

        return self.job("license", work, done)

    def do_open(self, a: dict) -> None:
        what = str(a.get("what"))
        url = {"support": release.SUPPORT_URL, "site": getattr(release, "SITE_URL", "")}.get(what, "")
        if url:
            webbrowser.open(url)
        elif what == "report" and self.ctrl.last_report():
            with contextlib.suppress(OSError):
                os.startfile(str(self.ctrl.last_report()))  # type: ignore[attr-defined]

    def shutdown(self) -> None:
        if self._hotkey is not None:
            self._hotkey.stop()
        self.ctrl.set_panic(False)
        if self.ctrl.engine.running:
            self.ctrl.stop()


# --- the local server ----------------------------------------------------------------------------------------
def make_handler(app: WebApp, token: str, seen: list[float]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _ok(self) -> bool:
            q = parse_qs(urlparse(self.path).query)
            if q.get("t", [""])[0] != token and self.headers.get("X-Token") != token:
                self._send(HTTPStatus.FORBIDDEN, b"{}", "application/json")
                return False
            seen[0] = time.monotonic()
            return True

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                if not self._ok():
                    return
                body = (STATIC / "index.html").read_bytes().replace(b"__TOKEN__", token.encode())
                self._send(200, body, TYPES[".html"])
            elif path.startswith("/static/"):
                f = (STATIC / path[len("/static/") :]).resolve()
                if STATIC.resolve() not in f.parents or not f.is_file():
                    self._send(404, b"", "text/plain")
                    return
                self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"), "max-age=60")
            elif path == "/api/state":
                if self._ok():
                    try:
                        body, code = json.dumps(app.state(), default=str).encode(), 200
                    except Exception as exc:
                        app.log(f"state failed:\n{traceback.format_exc()}")
                        body, code = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(), 500
                    self._send(code, body, "application/json")
            elif path == "/api/picture":
                if not self._ok():
                    return
                f = app.ctrl.pictures.file(parse_qs(urlparse(self.path).query).get("id", [""])[0])
                if f is None:
                    self._send(404, b"", "text/plain")
                else:
                    self._send(200, f.read_bytes(), "image/png")
            elif path == "/api/brand":
                if not self._ok():
                    return
                f = Path(app.ctrl.settings.brand_image) if app.ctrl.settings.brand_image else None
                if f is not None and f.is_file():
                    self._send(200, f.read_bytes(), "image/png")
                else:
                    self._send(404, b"", "text/plain")
            elif path == "/api/report_open":
                if self._ok():
                    app.do_open({"what": "report"})
                    self._send(200, b"{}", "application/json")
            elif path == "/api/preview":
                if not self._ok():
                    return
                f = app.ctrl.preview_file()
                if f.is_file():
                    self._send(200, f.read_bytes(), "image/png")
                else:
                    self._send(404, b"", "text/plain")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if not path.startswith("/api/") or not self._ok():
                return
            n = int(self.headers.get("Content-Length", "0") or 0)
            if path == "/api/brand_upload":  # the streamer's logo (Pro): raw image bytes
                out = app.brand_upload(self.rfile.read(n) if 0 < n <= 8_000_000 else b"")
                self._send(200, json.dumps(out, default=str).encode(), "application/json")
                return
            try:
                a = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
                out = app.act(path[len("/api/") :], a if isinstance(a, dict) else {})
            except Exception as exc:
                out = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
            self._send(200, json.dumps(out, default=str).encode(), "application/json")

    return Handler


def find_edge() -> str | None:
    for var in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if base:
            p = Path(base, "Microsoft", "Edge", "Application", "msedge.exe")
            if p.is_file():
                return str(p)
    return shutil.which("msedge")


def open_window(url: str, profile: Path, lang: str) -> subprocess.Popen | None:
    """The page in its own Edge window (app mode: no address bar, own profile). None: the default browser opened it."""
    edge = find_edge()
    if edge:
        return subprocess.Popen(
            [edge, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1280,820", "--no-first-run",
             "--no-default-browser-check", "--disable-features=Translate", "--lang=" + lang],
        )  # fmt: skip
    webbrowser.open(url)
    return None


def running_url(lock: Path) -> str:
    """The address of a window that is already open (a second start shows it instead of starting a second app)."""
    try:
        url = json.loads(lock.read_text(encoding="utf-8"))["url"]
        base, _, query = url.partition("/?")
        with urllib.request.urlopen(f"{base}/api/state?{query}", timeout=2) as r:
            return url if r.status == 200 else ""
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    probe = controller.Controller()
    folder = probe.preview_file().parent
    folder.mkdir(parents=True, exist_ok=True)
    lock, profile = folder / "window.json", folder / "window"
    url = running_url(lock)
    if url:
        open_window(url, profile, probe.settings.lang)
        return 0
    app = WebApp(probe)
    token = secrets.token_urlsafe(24)
    seen = [time.monotonic()]
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, token, seen))
    url = f"http://127.0.0.1:{server.server_address[1]}/?t={token}"
    threading.Thread(target=server.serve_forever, name="web-ui", daemon=True).start()
    lock.write_text(json.dumps({"url": url, "pid": os.getpid()}), encoding="utf-8")
    proc = open_window(url, profile, app.lang)
    try:
        while True:
            time.sleep(1.0)
            if proc is not None and proc.poll() is not None:
                break  # the window was closed
            if proc is None and time.monotonic() - seen[0] > 30:
                break  # the browser tab was closed (no polls for 30 s)
    except KeyboardInterrupt:
        pass
    app.shutdown()
    server.shutdown()
    with contextlib.suppress(OSError):
        lock.unlink()
    return 0
