"""The window. Arabic (right to left) is the default, English one button away.

Three tabs: Home (status, Start / Stop, what is still to set up, the last report), Settings (input from OBS, output
to the platform, the stream key) and Check (the system check in plain words, the NVIDIA fast mode, rollback).
All logic is in ``controller.py``; this file draws it and passes events. Worker threads never touch widgets: they
``post`` a function that the window runs on its own thread.

Right to left: text is right-aligned, a label sits at the RIGHT of its field, and in Arabic the tabs run from the
right. Tk itself shapes and orders Arabic text (checked on a real Windows runner, ``tools/ci_tk_arabic.py``).
Anything that is always left-to-right (an address, a key, a path) is left-aligned in its box.
"""

from __future__ import annotations

import contextlib
import os
import queue
import sys
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, messagebox, ttk

from streamsafe.app import build_info, controller, engine, i18n, modelpack, paths, release, updates
from streamsafe.app import settings as cfg
from streamsafe.branding import PRODUCT_NAME

FONT = "Segoe UI"
POLL_MS = 250
COLORS = {
    engine.STOPPED: "#5f6b7a",
    engine.STARTING: "#b26a00",
    engine.PROTECTED: "#1b7f3b",
    engine.COVERED: "#1f5fbf",
    engine.STALLED: "#b3261e",
    engine.FAILED: "#b3261e",
}
STATUS_KEYS = {
    engine.STOPPED: ("st_stopped", "st_hint_stopped"),
    engine.STARTING: ("st_starting", "st_hint_covered"),
    engine.PROTECTED: ("st_protected", "st_hint_protected"),
    engine.COVERED: ("st_covered", "st_hint_covered"),
    engine.STALLED: ("st_stalled", "st_hint_covered"),
    engine.FAILED: ("st_failed", "st_hint_covered"),
}
MARK = {"ok": ("●", "#1b7f3b"), "warn": ("▲", "#b26a00"), "fail": ("■", "#b3261e")}


def single_instance() -> bool:
    """False when another window of this program is already open on this PC (two engines would fight over the
    same port and state file). Windows only; elsewhere always True."""
    if sys.platform != "win32":
        return True
    import ctypes

    ctypes.windll.kernel32.CreateMutexW(None, False, f"{PRODUCT_NAME}-window")  # type: ignore[attr-defined]
    return ctypes.windll.kernel32.GetLastError() != 183  # type: ignore[attr-defined]  # ERROR_ALREADY_EXISTS


class App(tk.Tk):
    def __init__(self, ctrl: controller.Controller | None = None) -> None:
        super().__init__()
        self.ctrl = ctrl or controller.Controller(log=lambda line: self.post(lambda: self._log(line)))
        self.lang = self.ctrl.settings.lang
        self.title(PRODUCT_NAME)
        self.geometry("680x760")
        self.minsize(600, 640)
        self.configure(bg="#eef0f3")
        self.option_add("*Font", (FONT, 10))
        self._q: queue.Queue[Callable[[], None]] = queue.Queue()
        self._busy = threading.Event()  # a setup / update job is running
        self._cancel = threading.Event()
        self._last_status = engine.STOPPED
        self._announced_exit = False
        self._vars: dict[str, tk.Variable] = {}
        self._nvidia_offered = False
        self._banner_update = ""
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build()
        self.after(POLL_MS, self.tick)
        self.after(100, self.refresh_check)
        self.after(600, self.look_for_updates)

    # --- thread hand-over -----------------------------------------------------------------------------------
    def post(self, fn: Callable[[], None]) -> None:
        self._q.put(fn)

    def tick(self) -> None:
        with contextlib.suppress(queue.Empty):
            while True:
                self._q.get_nowait()()
        self.show_status()
        self.after(POLL_MS, self.tick)

    # --- text ---------------------------------------------------------------------------------------------
    def tr(self, key: str, **values: object) -> str:
        return i18n.t(self.lang, key, **values)

    @property
    def rtl(self) -> bool:
        return self.lang == "ar"

    @property
    def side(self) -> str:
        return "e" if self.rtl else "w"

    # --- building -----------------------------------------------------------------------------------------
    def build(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self._pull_vars()
        style = ttk.Style(self)
        with contextlib.suppress(tk.TclError):
            style.theme_use("vista" if sys.platform == "win32" else "clam")
        style.configure("Big.TButton", font=(FONT, 12, "bold"), padding=8)
        style.configure("TRadiobutton", background="white")
        style.configure("TCheckbutton", background="white")

        top = tk.Frame(self, bg="#eef0f3")
        top.pack(fill="x", padx=16, pady=(12, 0))
        lang = ttk.Button(top, text=self.tr("lang_button"), command=self.toggle_lang, width=9)
        title = tk.Label(top, text=self.tr("title"), font=(FONT, 18, "bold"), bg="#eef0f3", fg="#1d2733")
        ver = tk.Label(top, text=f"v{updates.current_version()}", font=(FONT, 9), bg="#eef0f3", fg="#6b7785")
        if self.rtl:
            lang.pack(side="left")
            title.pack(side="right")
            ver.pack(side="right", padx=8)
        else:
            lang.pack(side="right")
            title.pack(side="left")
            ver.pack(side="left", padx=8)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=16, pady=12)
        self.home, self.sets, self.chk = (tk.Frame(self.nb, bg="white", padx=14, pady=12) for _ in range(3))
        tabs = [(self.home, "tab_home"), (self.sets, "tab_settings"), (self.chk, "tab_check")]
        for frame, key in tabs[::-1] if self.rtl else tabs:  # Arabic: the first tab at the right
            self.nb.add(frame, text=self.tr(key))
        self.nb.select({"settings": self.sets, "check": self.chk}.get(os.environ.get("STREAMSAFE_TAB", ""), self.home))  # CI screenshots
        self.build_home()
        self.build_settings()
        self.build_check()
        self.show_check()
        self.show_status(force=True)

    def label(self, parent: tk.Misc, text: str = "", **kw) -> tk.Label:
        kw.setdefault("bg", "white")
        kw.setdefault("anchor", self.side)
        kw.setdefault("justify", "right" if self.rtl else "left")
        kw.setdefault("wraplength", 560)
        return tk.Label(parent, text=text, **kw)

    def pack_row(self, widget: tk.Widget, **kw) -> None:
        kw.setdefault("anchor", self.side)
        kw.setdefault("pady", 3)
        widget.pack(**kw)

    def var(self, name: str, kind: type[tk.Variable], value: object) -> tk.Variable:
        v = self._vars.get(name)
        if v is None:
            v = kind(self, value=value)  # type: ignore[call-arg]
            self._vars[name] = v
        return v

    def _pull_vars(self) -> None:
        """Keep what is typed when the window is rebuilt (language switch)."""
        if "input" in self._vars:
            self.collect()

    # --- home -----------------------------------------------------------------------------------------------
    def build_home(self) -> None:
        h = self.home
        self.badge = tk.Label(h, text="", font=(FONT, 24, "bold"), fg="white", bg=COLORS[engine.STOPPED], pady=14)
        self.badge.pack(fill="x")
        self.hint = self.label(h, "", fg="#44505e", pady=6)
        self.pack_row(self.hint, fill="x")
        self.banner = self.label(h, "", fg="#b26a00", font=(FONT, 10, "bold"))
        self.pack_row(self.banner, fill="x")
        self.update_row = tk.Frame(h, bg="white")
        self.update_text = self.label(self.update_row, "", fg="#1f5fbf")
        self.update_btn = ttk.Button(self.update_row, text=self.tr("update_button"), command=self.apply_update)
        self.setup_box = tk.LabelFrame(h, text=self.tr("sec_setup"), bg="white", padx=10, pady=8, labelanchor="ne" if self.rtl else "nw")
        self.setup_text = self.label(self.setup_box, "")
        self.setup_text.pack(fill="x")
        btns = tk.Frame(self.setup_box, bg="white")
        btns.pack(fill="x", pady=4)
        self.setup_buttons = [
            ttk.Button(btns, text=self.tr("download_models"), command=lambda: self.install("download")),
            ttk.Button(btns, text=self.tr("choose_zip"), command=lambda: self.install("zip")),
            ttk.Button(btns, text=self.tr("choose_folder"), command=lambda: self.install("folder")),
        ]
        for b in self.setup_buttons if not self.rtl else self.setup_buttons[::-1]:
            b.pack(side="right" if self.rtl else "left", padx=3)
        self.progress = ttk.Progressbar(self.setup_box, mode="determinate", maximum=100)
        self.progress_text = self.label(self.setup_box, "", fg="#44505e")
        self.cancel_btn = ttk.Button(self.setup_box, text=self.tr("cancel"), command=self._cancel.set)

        row = tk.Frame(h, bg="white")
        row.pack(fill="x", pady=(14, 4))
        self.start_btn = ttk.Button(row, text=self.tr("start"), style="Big.TButton", command=self.on_start)
        self.stop_btn = ttk.Button(row, text=self.tr("stop"), style="Big.TButton", command=self.on_stop)
        for b in (self.start_btn, self.stop_btn):
            b.pack(side="right" if self.rtl else "left", padx=4, expand=True, fill="x")
        self.message = self.label(h, "", fg="#b3261e", font=(FONT, 10, "bold"))
        self.pack_row(self.message, fill="x")
        self.report_btn = ttk.Button(h, text=self.tr("open_report"), command=self.open_report)
        self.pack_row(self.report_btn, pady=(10, 2))
        self.logbox = tk.Label(h, text="", bg="white", fg="#8a94a1", font=(FONT, 8), anchor="w", justify="left", wraplength=580)
        self.logbox.pack(fill="x", side="bottom")

    # --- settings -------------------------------------------------------------------------------------------
    def build_settings(self) -> None:
        s, st = self.sets, self.ctrl.settings
        v = self.var
        v("input", tk.StringVar, st.input).set(st.input)
        v("srt_port", tk.StringVar, str(st.srt_port))
        v("camera", tk.StringVar, st.camera)
        v("output", tk.StringVar, st.output)
        v("rtmp_url", tk.StringVar, st.rtmp_url)
        v("key", tk.StringVar, self.ctrl.stored_key() if "key" not in self._vars else self._vars["key"].get())
        v("remember", tk.BooleanVar, st.remember_key)
        v("delay", tk.StringVar, f"{st.delay:g}")

        def field(parent: tk.Misc, caption: str, var: tk.Variable, width: int = 34, show: str = "") -> None:
            fr = tk.Frame(parent, bg="white")
            lab = self.label(fr, caption, wraplength=200)
            ent = tk.Entry(fr, textvariable=var, width=width, justify="left", show=show, relief="solid", bd=1)
            if self.rtl:
                lab.pack(side="right", padx=(8, 0))
                ent.pack(side="right", fill="x", expand=True)
            else:
                lab.pack(side="left", padx=(0, 8))
                ent.pack(side="left", fill="x", expand=True)
            self.pack_row(fr, fill="x")

        self.label(s, self.tr("sec_in"), font=(FONT, 11, "bold")).pack(fill="x")
        self.pack_row(ttk.Radiobutton(s, text=self.tr("in_srt"), value=cfg.INPUT_SRT, variable=v("input", tk.StringVar, "")))
        field(s, self.tr("srt_port"), v("srt_port", tk.StringVar, ""), 8)
        self.srt_hint = self.label(s, "", fg="#6b7785", font=(FONT, 9))
        self.pack_row(self.srt_hint, fill="x")
        self.pack_row(ttk.Radiobutton(s, text=self.tr("in_camera"), value=cfg.INPUT_CAMERA, variable=v("input", tk.StringVar, "")))
        field(s, self.tr("camera_name"), v("camera", tk.StringVar, ""))
        ttk.Separator(s).pack(fill="x", pady=10)
        self.label(s, self.tr("sec_out"), font=(FONT, 11, "bold")).pack(fill="x")
        self.pack_row(ttk.Radiobutton(s, text=self.tr("out_window"), value=cfg.OUTPUT_WINDOW, variable=v("output", tk.StringVar, "")))
        self.pack_row(ttk.Radiobutton(s, text=self.tr("out_rtmp"), value=cfg.OUTPUT_RTMP, variable=v("output", tk.StringVar, "")))
        field(s, self.tr("rtmp_url"), v("rtmp_url", tk.StringVar, ""))
        field(s, self.tr("stream_key"), v("key", tk.StringVar, ""), show="•")
        self.pack_row(ttk.Checkbutton(s, text=self.tr("remember_key"), variable=v("remember", tk.BooleanVar, False)))
        if not self.ctrl.keys.available:
            self.pack_row(self.label(s, self.tr("key_no_store"), fg="#6b7785", font=(FONT, 9)), fill="x")
        field(s, self.tr("delay"), v("delay", tk.StringVar, ""), 8)
        ttk.Separator(s).pack(fill="x", pady=10)
        self.label(s, self.tr("open_obs"), font=(FONT, 11, "bold")).pack(fill="x")
        for key, path in (("obs_state", paths.obs_state_file(self.ctrl.env)), ("obs_script", paths.obs_script())):
            fr = tk.Frame(s, bg="white")
            ttk.Button(fr, text=self.tr("copy"), width=8, command=lambda p=str(path): self.copy(p)).pack(
                side="left" if not self.rtl else "right", padx=4
            )
            tk.Label(
                fr, text=self.tr(key, path=path), bg="white", fg="#44505e", font=(FONT, 8), anchor="w", justify="left", wraplength=470
            ).pack(side="right" if self.rtl else "left", fill="x", expand=True)
            self.pack_row(fr, fill="x")
        v("input", tk.StringVar, "").trace_add("write", lambda *_: self.show_srt_hint())
        v("srt_port", tk.StringVar, "").trace_add("write", lambda *_: self.show_srt_hint())
        self.show_srt_hint()

    def show_srt_hint(self) -> None:
        port = self._vars["srt_port"].get().strip() or "9000"
        self.srt_hint.configure(text=self.tr("srt_hint", port=port))

    def collect(self) -> None:
        """Window fields -> settings (values that are not valid stay as typed so that ``problems`` reports them)."""
        st, v = self.ctrl.settings, self._vars
        st.lang = self.lang
        st.input, st.output = str(v["input"].get()), str(v["output"].get())
        st.camera, st.rtmp_url = str(v["camera"].get()).strip(), str(v["rtmp_url"].get()).strip()
        st.remember_key = bool(v["remember"].get())
        try:
            st.srt_port = int(str(v["srt_port"].get()).strip())
        except ValueError:
            st.srt_port = 0
        try:
            st.delay = float(str(v["delay"].get()).strip().replace(",", "."))
        except ValueError:
            st.delay = 0.0

    # --- check tab ------------------------------------------------------------------------------------------
    def build_check(self) -> None:
        c = self.chk
        top = tk.Frame(c, bg="white")
        top.pack(fill="x")
        self.verdict = self.label(top, "", font=(FONT, 11, "bold"), wraplength=380)
        self.recheck_btn = ttk.Button(top, text=self.tr("recheck"), command=self.refresh_check)
        if self.rtl:
            self.recheck_btn.pack(side="left")
            self.verdict.pack(side="right", fill="x", expand=True)
        else:
            self.recheck_btn.pack(side="right")
            self.verdict.pack(side="left", fill="x", expand=True)
        self.check_list = tk.Frame(c, bg="white")
        self.check_list.pack(fill="x", pady=8)
        self.nvidia_box = tk.Frame(c, bg="white")
        self.nvidia_text = self.label(self.nvidia_box, "")
        self.nvidia_text.pack(fill="x")
        nb = tk.Frame(self.nvidia_box, bg="white")
        nb.pack(fill="x", pady=4)
        for text, cmd in ((self.tr("nvidia_button"), self.enable_nvidia), (self.tr("nvidia_later"), self.decline_nvidia)):
            ttk.Button(nb, text=text, command=cmd).pack(side="right" if self.rtl else "left", padx=3)
        self.rollback_btn = ttk.Button(c, text=self.tr("rollback"), command=self.roll_back)

    def show_check(self) -> None:
        r = self.ctrl.check_result
        for child in self.check_list.winfo_children():
            child.destroy()
        if not r.verdict:
            self.verdict.configure(text=self.tr("check_failed", why=r.error) if r.error else self.tr("checking"))
        else:
            self.verdict.configure(text=self.tr(f"verdict_{r.verdict}"))
        for item in r.checks:
            mark, color = MARK.get(item.get("status", ""), ("?", "#6b7785"))
            name = item.get("name", "")
            fr = tk.Frame(self.check_list, bg="white")
            head = f"{mark}  {self.tr('chk_' + name) if 'chk_' + name in i18n.TEXT else name}: {item.get('detail', '')}"
            self.label(fr, head, fg=color, font=(FONT, 10, "bold"), wraplength=560).pack(fill="x")
            fix = item.get("fix_ar" if self.rtl else "fix_en", "")
            if fix and item.get("status") != "ok":
                self.label(fr, fix, fg="#44505e", font=(FONT, 9), wraplength=540).pack(fill="x")
            self.pack_row(fr, fill="x", pady=2)
        self.show_setup()

    def show_setup(self) -> None:
        problem = self.ctrl.models_problem()
        if problem:
            self.setup_text.configure(text=self.tr("models_needed") if "not installed" in problem else self.tr("models_bad", why=problem))
            self.setup_box.pack(fill="x", pady=8, before=self.start_btn.master)
        else:
            self.setup_box.pack_forget()
        offer = (
            not problem
            and self.ctrl.has_nvidia_gpu()
            and not self.ctrl.use_nvidia
            and not self.ctrl.settings.nvidia_declined
            and not self.ctrl.engine.running
        )
        if offer:
            self.nvidia_text.configure(text=self.tr("nvidia_offer", gb=release.NVIDIA_DOWNLOAD_GB))
            self.nvidia_box.pack(fill="x", pady=8)
        else:
            self.nvidia_box.pack_forget()
        if self.ctrl.can_roll_back():
            self.rollback_btn.pack(anchor=self.side, pady=6)
        else:
            self.rollback_btn.pack_forget()

    def refresh_check(self) -> None:
        if self._busy.is_set():
            return
        self.ctrl.check_result = controller.CheckResult()
        self.show_check()

        def work() -> None:
            self.ctrl.run_check()
            self.post(self.show_check)

        threading.Thread(target=work, name="check", daemon=True).start()

    # --- status ---------------------------------------------------------------------------------------------
    def show_status(self, force: bool = False) -> None:
        status = self.ctrl.status()
        if status == self._last_status and not force:
            self.update_controls()
            return
        self._last_status = status
        label, hint = STATUS_KEYS[status]
        if status == engine.STOPPED and self.ctrl.expired():
            label, hint = "st_blocked", "beta_expired"
        self.badge.configure(text=self.tr(label), bg=COLORS[status])
        self.hint.configure(text=self.tr(hint))
        if status in (engine.STOPPED, engine.FAILED) and self.ctrl.last_exit not in (None, 0) and not self._announced_exit:
            self._announced_exit = True
            self.message.configure(text=self.tr(self.ctrl.exit_reason(), code=self.ctrl.last_exit))
        self.update_controls()
        left = self.ctrl.days_left()
        if left is None:
            self.banner.configure(text="")
        elif left <= 0:
            self.banner.configure(text=self.tr("beta_expired"), fg="#b3261e")
        else:
            date = build_info.EXPIRES
            self.banner.configure(text=self.tr("beta_days" if left > 1 else "beta_last_day", days=left, date=date), fg="#b26a00")

    def update_controls(self) -> None:
        running = self.ctrl.engine.running
        busy = self._busy.is_set()
        can = not running and not busy and not self.ctrl.expired()
        self.start_btn.state(["!disabled"] if can else ["disabled"])
        self.stop_btn.state(["!disabled"] if running else ["disabled"])
        for b in self.setup_buttons:
            b.state(["disabled"] if busy or running else ["!disabled"])
        self.update_btn.state(["disabled"] if busy or running else ["!disabled"])

    # --- actions --------------------------------------------------------------------------------------------
    def toggle_lang(self) -> None:
        self.collect()
        self.lang = "en" if self.rtl else "ar"
        self.ctrl.settings.lang = self.lang
        self.ctrl.save_settings()
        self._last_status = ""
        self.build()
        self.show_update_row()

    def _log(self, line: str) -> None:
        self.logbox.configure(text=line[:160])

    def copy(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)

    def open_report(self) -> None:
        path = self.ctrl.last_report()
        if path is None:
            self.message.configure(text=self.tr("no_report"), fg="#44505e")
            return
        with contextlib.suppress(OSError):
            engine.open_file(path)

    def on_start(self) -> None:
        self.collect()
        key = str(self._vars["key"].get()).strip()
        self.message.configure(text="", fg="#b3261e")
        self._announced_exit = False
        why = self.ctrl.blockers(key)
        if why:
            self.message.configure(text=self.tr("not_ready", why=self.tr(why[0])))
            return
        if not self.ctrl.remember(key):
            self._vars["remember"].set(False)
        self.ctrl.save_settings()
        try:
            self.ctrl.start(key)
        except (RuntimeError, OSError) as exc:
            self.message.configure(text=self.tr("not_ready", why=self.tr(str(exc)) if str(exc) in i18n.TEXT else str(exc)))
        self._last_status = ""
        self.show_status(force=True)

    def on_stop(self) -> None:
        self.message.configure(text=self.tr("stopping"), fg="#44505e")
        self.stop_btn.state(["disabled"])

        def work() -> None:
            self.ctrl.stop()
            self.post(lambda: self.message.configure(text=""))

        threading.Thread(target=work, name="stop", daemon=True).start()

    def job(self, name: str, work: Callable[[], str], done: Callable[[str], None] | None = None) -> None:
        """Run ``work`` (returns an error text or "") on a worker thread with the progress bar shown."""
        if self._busy.is_set():
            return
        self._busy.set()
        self._cancel.clear()
        self.nb.select(self.home)
        self.progress.pack(fill="x", pady=4)
        self.progress_text.pack(fill="x")
        self.cancel_btn.pack(anchor=self.side)
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.progress_text.configure(text=self.tr("installing"))
        self.setup_box.pack(fill="x", pady=8, before=self.start_btn.master)

        def run() -> None:
            try:
                err = work()
            except Exception as exc:  # a worker must never die silently: show it
                err = f"{type(exc).__name__}: {exc}"
            self.post(lambda: self.job_done(err, done))

        threading.Thread(target=run, name=name, daemon=True).start()

    def job_done(self, err: str, done: Callable[[str], None] | None) -> None:
        self._busy.clear()
        self.progress.stop()
        for w in (self.progress, self.progress_text, self.cancel_btn):
            w.pack_forget()
        self.message.configure(text=err, fg="#b3261e" if err else "#1b7f3b")
        if done:
            done(err)
        self.refresh_check()

    def on_progress(self, what: str, done: int, total: int) -> None:
        def show() -> None:
            word = self.tr("p_" + what) if "p_" + what in i18n.TEXT else what
            if total > 0:
                self.progress.stop()
                self.progress.configure(mode="determinate", value=100 * min(done, total) / total)
                size = f" {done / 1e6:,.0f} / {total / 1e6:,.0f} MB" if total > 5_000_000 else f" {done}/{total}"
                self.progress_text.configure(text=word + size)
            else:
                self.progress_text.configure(text=word)

        self.post(show)

    def install(self, source: str) -> None:
        location = ""
        if source == "zip":
            location = filedialog.askopenfilename(title=self.tr("pick_zip_title"), filetypes=[("zip", "*.zip")])
        elif source == "folder":
            location = filedialog.askdirectory(title=self.tr("pick_folder_title"))
        if source != "download" and not location:
            return
        self.message.configure(text="")

        def work() -> str:
            try:
                self.ctrl.install_models(source, location, self.on_progress, self._cancel)
            except modelpack.SourceError as exc:
                return self.tr(f"src_{exc.kind}", why=str(exc))
            except modelpack.PackError as exc:
                return self.tr("pack_bad", why=str(exc))
            except (OSError, InterruptedError):
                return self.tr("install_cancelled") if self._cancel.is_set() else self.tr("src_network", why="disk")
            except Exception as exc:  # fetch.CancelledError and anything unexpected: nothing was changed
                return self.tr("install_cancelled") if self._cancel.is_set() else self.tr("src_other", why=str(exc))
            return ""

        self.job(
            "models", work, lambda err: self.message.configure(text=err or self.tr("install_done"), fg="#b3261e" if err else "#1b7f3b")
        )

    def enable_nvidia(self) -> None:
        def work() -> str:
            why = self.ctrl.enable_nvidia(
                lambda w, d, t: self.on_progress("nvidia_" + ("downloading" if t > 1000 else "preparing"), d, t), self._cancel
            )
            return self.tr("nvidia_failed", why=why) if why else ""

        self.job(
            "nvidia", work, lambda err: self.message.configure(text=err or self.tr("nvidia_ready"), fg="#b3261e" if err else "#1b7f3b")
        )

    def decline_nvidia(self) -> None:
        self.ctrl.settings.nvidia_declined = True
        self.ctrl.save_settings()
        self.show_setup()

    # --- updates --------------------------------------------------------------------------------------------
    def look_for_updates(self) -> None:
        def work() -> None:
            if self.ctrl.check_updates() is not None:
                self.post(self.show_update_row)

        threading.Thread(target=work, name="update-check", daemon=True).start()

    def show_update_row(self) -> None:
        app_new, data_new = self.ctrl.update_wanted()
        self.update_row.pack_forget()
        if not (app_new or data_new):
            return
        parts = []
        if app_new and self.ctrl.update:
            parts.append(self.tr("update_app", version=self.ctrl.update.app_version))
        if data_new:
            parts.append(self.tr("update_data"))
        self.update_text.configure(text="  ".join(parts))
        for w in self.update_row.winfo_children():
            w.pack_forget()
        side = "right" if self.rtl else "left"
        self.update_text.pack(side=side, fill="x", expand=True)
        if data_new:
            self.update_btn.pack(side="left" if self.rtl else "right", padx=6)
        self.update_row.pack(fill="x", pady=4, before=self.setup_box if self.setup_box.winfo_ismapped() else self.start_btn.master)

    def apply_update(self) -> None:
        if self.ctrl.engine.running:
            self.message.configure(text=self.tr("update_busy"))
            return

        def work() -> str:
            try:
                self.ctrl.apply_update(lambda w, d, t: self.on_progress(w, d, t), self._cancel)
            except updates.UpdateError as exc:
                return self.tr("update_failed", why=str(exc))
            except Exception as exc:
                return self.tr("update_failed", why=str(exc)) if not self._cancel.is_set() else self.tr("install_cancelled")
            return ""

        self.job(
            "update",
            work,
            lambda err: (
                self.message.configure(text=err or self.tr("update_done"), fg="#b3261e" if err else "#1b7f3b"),
                self.show_update_row(),
            ),
        )

    def roll_back(self) -> None:
        try:
            self.ctrl.roll_back()
            self.message.configure(text=self.tr("rollback_done"), fg="#1b7f3b")
        except modelpack.PackError as exc:
            self.message.configure(text=str(exc), fg="#b3261e")
        self.refresh_check()

    # --- closing --------------------------------------------------------------------------------------------
    def on_close(self) -> None:
        if self.ctrl.engine.running:
            if not messagebox.askyesno(PRODUCT_NAME, self.tr("quit_running")):
                return
            self.ctrl.stop()  # clean stop: outputs closed, report written; killed after 20 s if it does not end
        self._cancel.set()
        self.destroy()


def main(argv: list[str] | None = None) -> int:
    if not single_instance():
        return 0
    app = App()
    app.mainloop()
    return 0
