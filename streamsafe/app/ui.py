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
import webbrowser
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from streamsafe.app import (
    build_info,
    controller,
    engine,
    hotkey,
    i18n,
    ingest,
    modelpack,
    obs_setup,
    passkey,
    paths,
    plan,
    preflight,
    release,
    updates,
)
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
# The palette of the logo and the website: night indigo, the navy of the cover cloth, dusk pink, warm sand.
NIGHT, COVER, DUSK, LAGOON = "#120b2b", "#1e2a52", "#ff4fa3", "#0e8ea0"
BLUSH = "#ffeef6"  # the support card
SAND, SAND2, CARD, LINE, INK, MUTED = "#f4ede2", "#ebe1d1", "#ffffff", "#e6dccb", "#1b1530", "#6d6380"


class Tile(tk.Frame):
    """A big coloured button with a title and one line under it (like a card). ``state`` follows ttk's
    ``["disabled"]`` / ``["!disabled"]`` so the rest of the window treats it like any button."""

    def __init__(self, parent: tk.Misc, title: str, sub: str, bg: str, fg: str, command: Callable[[], None]) -> None:
        super().__init__(parent, bg=bg, padx=16, pady=12, cursor="hand2")
        self._bg, self._fg, self._command, self._disabled = bg, fg, command, False
        self._title = tk.Label(self, text=title, font=(FONT, 15, "bold"), bg=bg, fg=fg)
        self._sub = tk.Label(self, text=sub, font=(FONT, 9), bg=bg, fg=fg)
        self._title.pack()
        self._sub.pack()
        for w in (self, self._title, self._sub):
            w.bind("<Button-1>", lambda e: self.invoke())

    def invoke(self) -> None:
        if not self._disabled:
            self._command()

    def state(self, spec: list[str] | None = None) -> tuple[str, ...]:
        if spec:
            self._disabled = "disabled" in spec
            bg, fg = ("#d9d2c7", "#8f8799") if self._disabled else (self._bg, self._fg)
            for w in (self, self._title, self._sub):
                w.configure(bg=bg)
            for w in (self._title, self._sub):
                w.configure(fg=fg)
            self.configure(cursor="arrow" if self._disabled else "hand2")
        return ("disabled",) if self._disabled else ()

    def instate(self, spec: list[str]) -> bool:
        return ("disabled" in spec) == self._disabled


def single_instance() -> bool:
    """False when another window of this program is already open on this PC (two engines would fight over the
    same port and state file). Windows only; elsewhere always True."""
    if sys.platform != "win32":
        return True
    import ctypes

    ctypes.windll.kernel32.CreateMutexW(None, False, f"{PRODUCT_NAME}-window")  # type: ignore[attr-defined]
    return ctypes.windll.kernel32.GetLastError() != 183  # type: ignore[attr-defined]  # ERROR_ALREADY_EXISTS


RLE, PDF = "\u202b", "\u202c"


class App(tk.Tk):
    def __init__(self, ctrl: controller.Controller | None = None) -> None:
        super().__init__()
        self.ctrl = ctrl or controller.Controller(log=lambda line: self.post(lambda: self._log(line)))
        self.lang = self.ctrl.settings.lang
        self.title(PRODUCT_NAME)
        self._set_icon()
        self.geometry("700x820")
        self.minsize(620, 660)
        self.configure(bg=SAND)
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
        self._hotkey: hotkey.Hotkey | None = None
        self.build()
        self.arm_panic_key()
        self.after(POLL_MS, self.tick)
        self.after(100, self.refresh_check)
        self.after(600, self.look_for_updates)
        threading.Thread(target=self.sync_pass_clock, daemon=True).start()

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
        text = i18n.t(self.lang, key, **values)
        # Tk 8.6 on Windows lays paragraphs out left-to-right, which puts Arabic punctuation on the wrong
        # side; an explicit right-to-left embedding fixes the base direction.
        return f"{RLE}{text}{PDF}" if self.rtl else text

    @property
    def rtl(self) -> bool:
        return self.lang == "ar"

    @property
    def side(self) -> str:
        return "e" if self.rtl else "w"

    # --- building -----------------------------------------------------------------------------------------
    def _set_icon(self) -> None:
        """Our logo in the title bar and taskbar (the exe itself gets it from the build)."""
        brand = paths.bundled_brand_dir()
        with contextlib.suppress(tk.TclError, OSError):
            if sys.platform == "win32" and (brand / "icon.ico").is_file():
                self.iconbitmap(default=str(brand / "icon.ico"))
            elif (brand / "icon_small.png").is_file():
                self._icon_img = tk.PhotoImage(file=str(brand / "icon_small.png"))
                self.iconphoto(True, self._icon_img)

    def build(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self._pull_vars()
        self.configure(bg=SAND)
        style = ttk.Style(self)
        with contextlib.suppress(tk.TclError):
            style.theme_use("clam")  # the same flat look on every Windows version; "vista" ignores our colours
        style.configure(".", background=CARD, foreground=INK, font=(FONT, 10))
        style.configure("TNotebook", background=SAND, borderwidth=0, tabmargins=(0, 6, 0, 0))
        style.configure("TNotebook.Tab", background=SAND, foreground=MUTED, padding=(18, 8), font=(FONT, 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", INK)])
        style.configure("TButton", background=CARD, foreground=INK, bordercolor="#d6c9b4", lightcolor=CARD, darkcolor=CARD, padding=(12, 5))
        bar = {"background": SAND2, "troughcolor": CARD, "bordercolor": CARD, "arrowcolor": MUTED, "lightcolor": SAND2, "darkcolor": SAND2}
        style.configure("Vertical.TScrollbar", **bar)
        style.map("TButton", background=[("active", SAND2), ("disabled", CARD)], foreground=[("disabled", "#b5adbf")])
        style.configure("TRadiobutton", background=CARD, foreground=INK)
        style.configure("TCheckbutton", background=CARD, foreground=INK)
        style.configure("TCombobox", fieldbackground="white", background=CARD, bordercolor=LINE, arrowsize=14, padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", "white")], foreground=[("readonly", INK)])
        style.configure("Horizontal.TProgressbar", background=DUSK, troughcolor=SAND2, bordercolor=SAND2)

        top = tk.Frame(self, bg=NIGHT)
        top.pack(fill="x")
        inner = tk.Frame(top, bg=NIGHT)
        inner.pack(fill="x", padx=16, pady=10)
        self._logo = None
        with contextlib.suppress(tk.TclError, OSError):
            small = paths.bundled_brand_dir() / "icon_small.png"
            if small.is_file():
                self._logo = tk.PhotoImage(file=str(small)).subsample(2, 2)  # 64 px -> 32 px
        logo = tk.Label(inner, image=self._logo, bg=NIGHT) if self._logo else None
        title = tk.Label(inner, text=PRODUCT_NAME, font=(FONT, 15, "bold"), bg=NIGHT, fg="white")
        ver = tk.Label(inner, text=f"v{updates.current_version()}", font=(FONT, 9), bg=NIGHT, fg="#9b8fbf")
        lang = tk.Label(
            inner, text=self.tr("lang_button"), font=(FONT, 10, "bold"), bg="#2a1f4d", fg="white", padx=12, pady=4, cursor="hand2"
        )
        lang.bind("<Button-1>", lambda e: self.toggle_lang())
        near, far = ("right", "left") if self.rtl else ("left", "right")
        for w in (logo, title, ver):
            if w is not None:
                w.pack(side=near, padx=(0, 8) if near == "left" else (8, 0))
        lang.pack(side=far)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=14, pady=(10, 14))
        self.home, self.sets, self.chk = (tk.Frame(self.nb, bg=CARD, padx=16, pady=14) for _ in range(3))
        tabs = [(self.home, "tab_home"), (self.sets, "tab_settings"), (self.chk, "tab_check")]
        for frame, key in tabs[::-1] if self.rtl else tabs:  # Arabic: the first tab at the right
            self.nb.add(frame, text=self.tr(key))
        self.nb.select({"settings": self.sets, "check": self.chk}.get(os.environ.get("STREAMSAFE_TAB", ""), self.home))  # CI screenshots
        self.prepare_vars()
        self.build_home()
        self.build_settings()
        self.build_check()
        self.show_check()
        self.show_status(force=True)

    def label(self, parent: tk.Misc, text: str = "", **kw) -> tk.Label:
        kw.setdefault("bg", CARD)
        kw.setdefault("fg", INK)
        kw.setdefault("anchor", self.side)
        kw.setdefault("justify", "right" if self.rtl else "left")
        kw.setdefault("wraplength", 560)
        return tk.Label(parent, text=text, **kw)

    def collapsing(self, parent: tk.Misc, **kw) -> tk.Label:
        """A message line that takes no room while it is empty (its own holder keeps its place in the window)."""
        holder = tk.Frame(parent, bg=CARD, height=1)
        holder.pack(fill="x")
        lab = self.label(holder, "", **kw)
        plain = lab.configure

        def configure(cnf=None, **opts):  # type: ignore[no-untyped-def]
            out = plain(cnf, **opts)
            if "text" in opts:
                if opts["text"]:
                    self.pack_row(lab, fill="x")
                else:
                    lab.pack_forget()
            return out

        lab.configure = lab.config = configure  # type: ignore[method-assign]
        return lab

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

    def prepare_vars(self) -> None:
        st, v = self.ctrl.settings, self.var
        v("input", tk.StringVar, st.input).set(st.input)
        v("srt_port", tk.StringVar, str(st.srt_port))
        v("camera", tk.StringVar, st.camera)
        v("camera_sound", tk.StringVar, st.camera_sound)
        v("input_file", tk.StringVar, st.input_file)
        v("output", tk.StringVar, st.output)
        v("platform", tk.StringVar, "").set(self.tr("plat_" + st.platform))  # the name in this language
        v("rtmp_url", tk.StringVar, st.rtmp_url)
        v("record_dir", tk.StringVar, st.record_dir)
        v("key", tk.StringVar, self.ctrl.stored_key() if "key" not in self._vars else self._vars["key"].get())
        v("remember", tk.BooleanVar, st.remember_key)
        v("delay", tk.StringVar, "").set(self.tr("delay_auto") if st.delay <= 0 else f"{st.delay:g}")
        v("resolution", tk.StringVar, "").set(st.resolution)
        v("fps", tk.StringVar, "").set(str(st.fps))
        v("translate", tk.BooleanVar, st.translate_ar)
        v("follow_obs", tk.BooleanVar, st.follow_obs)
        v("cover_mode", tk.StringVar, st.cover_mode)
        v("sensitivity", tk.StringVar, st.sensitivity)
        v("platform2", tk.StringVar, "").set(self.tr("plat_" + st.platform2) if st.platform2 else self.tr("plat2_none"))
        v("rtmp_url2", tk.StringVar, st.rtmp_url2)
        v("key2", tk.StringVar, self.ctrl.stored_key2() if "key2" not in self._vars else self._vars["key2"].get())
        v("panic_key", tk.StringVar, st.panic_button)
        v("license_key", tk.StringVar, "")

    # --- small building blocks -------------------------------------------------------------------------------
    def card(self, parent: tk.Misc, title: str = "", note: str = "") -> tk.Frame:
        """A white box with a thin border, a bold title and an optional one-line note under it."""
        outer = tk.Frame(parent, bg=LINE, padx=1, pady=1)
        outer.pack(fill="x", pady=(0, 12))
        box = tk.Frame(outer, bg=CARD, padx=14, pady=12)
        box.pack(fill="x")
        if title:
            self.label(box, title, font=(FONT, 12, "bold")).pack(fill="x")
        if note:
            self.label(box, note, fg=MUTED, font=(FONT, 9)).pack(fill="x", pady=(2, 6))
        return box

    def step(self, parent: tk.Misc, number: int, title: str) -> tk.Frame:
        """A numbered step: a pink number, a title, and a frame under them for the fields."""
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=(10, 2))
        side = "right" if self.rtl else "left"
        tk.Label(row, text=str(number), font=(FONT, 10, "bold"), bg=DUSK, fg="white", width=2).pack(side=side)
        self.label(row, title, font=(FONT, 11, "bold")).pack(side=side, padx=8)
        body = tk.Frame(parent, bg=CARD)
        body.pack(fill="x", padx=(0, 30) if self.rtl else (30, 0))
        return body

    def field(
        self, parent: tk.Misc, caption: str, var: tk.Variable, width: int = 34, show: str = "", combo: list[str] | None = None
    ) -> tk.Widget:
        side = "right" if self.rtl else "left"
        fr = tk.Frame(parent, bg=CARD)
        lab = self.label(fr, caption, wraplength=180, width=14 if width >= 20 else 0)
        if combo is not None:
            ent: tk.Widget = ttk.Combobox(fr, textvariable=var, values=combo, width=width)
        else:
            ent = tk.Entry(
                fr, textvariable=var, width=width, justify="left", show=show, relief="flat", bd=0, bg="white",
                highlightthickness=1, highlightbackground=LINE, highlightcolor=DUSK, font=(FONT, 10),
            )  # fmt: skip
        lab.pack(side=side, padx=(8, 0) if self.rtl else (0, 8))
        if width >= 20:
            ent.pack(side=side, fill="x", expand=True, ipady=4)
        else:
            ent.pack(side=side, ipady=3)
        self.pack_row(fr, fill="x", pady=4)
        return ent

    def small(self, parent: tk.Misc, text: str) -> tk.Label:
        lab = self.label(parent, text, fg=MUTED, font=(FONT, 9), wraplength=500)
        self.pack_row(lab, fill="x", pady=(0, 4))
        return lab

    def buttons(self, parent: tk.Misc, *items: tuple[str, Callable[[], None]]) -> tk.Frame:
        side = "right" if self.rtl else "left"
        fr = tk.Frame(parent, bg=CARD)
        for text, cmd in items:
            ttk.Button(fr, text=text, command=cmd).pack(side=side, padx=(6, 0) if self.rtl else (0, 6))
        self.pack_row(fr, fill="x", pady=4)
        return fr

    # --- home: status, the two big buttons, and the whole setup in three steps -----------------------------
    def build_home(self) -> None:
        h = self.scroll_area(self.home)
        side = "right" if self.rtl else "left"
        self.badge = tk.Label(h, text="", font=(FONT, 20, "bold"), fg="white", bg=COLORS[engine.STOPPED], pady=12)
        self.badge.pack(fill="x")
        self.hint = self.label(h, "", fg=MUTED, pady=6)
        self.pack_row(self.hint, fill="x")
        self.banner = self.collapsing(h, fg="#b26a00", font=(FONT, 10, "bold"))
        self.update_row = tk.Frame(h, bg=CARD)
        self.update_text = self.label(self.update_row, "", fg=LAGOON)
        self.update_btn = ttk.Button(self.update_row, text=self.tr("update_button"), command=self.apply_update)
        self.app_update_btn = ttk.Button(self.update_row, text=self.tr("update_app_now"), command=self.apply_app_update)

        row = tk.Frame(h, bg=CARD)
        row.pack(fill="x", pady=(6, 4))
        self.start_btn = Tile(row, self.tr("start"), self.tr("start_sub"), DUSK, "#1a0716", self.on_start)
        self.stop_btn = Tile(row, self.tr("stop"), self.tr("stop_sub"), COVER, "white", self.on_stop)
        for b in (self.start_btn, self.stop_btn):
            b.pack(side=side, padx=(0, 6), expand=True, fill="x")
        prow = tk.Frame(h, bg=CARD)
        prow.pack(fill="x", pady=(0, 4))
        self.preview_btn = ttk.Button(prow, text=self.tr("preview"), command=lambda: self.on_start(preview=True))
        self.preview_btn.pack(side=side)
        self.message = self.collapsing(h, fg="#b3261e", font=(FONT, 10, "bold"))
        self.pre_note = self.collapsing(h, fg="#b26a00", font=(FONT, 9, "bold"))
        # the two links while it runs: OBS -> us, us -> the platform (or the recording)
        self.link_box = tk.Frame(h, bg=SAND, padx=12, pady=8)
        self.lk_in = self.label(self.link_box, "", bg=SAND, font=(FONT, 10, "bold"), wraplength=520)
        self.lk_in.pack(fill="x")
        self.lk_out = self.label(self.link_box, "", bg=SAND, font=(FONT, 10, "bold"), wraplength=520)
        self.lk_out.pack(fill="x")
        self.lk_copy = ttk.Button(self.link_box, text=self.tr("copy_server"), command=lambda: self.copy(self.ctrl.obs_server()))
        self._link_shown: tuple = ()

        # first run: the models
        self.setup_box = tk.Frame(h, bg=SAND, padx=12, pady=10)
        self.setup_text = self.label(self.setup_box, "", bg=SAND)
        self.setup_text.pack(fill="x")
        btns = tk.Frame(self.setup_box, bg=SAND)
        btns.pack(fill="x", pady=4)
        self.setup_buttons = [
            ttk.Button(btns, text=self.tr("download_models"), command=lambda: self.install("download")),
            ttk.Button(btns, text=self.tr("choose_zip"), command=lambda: self.install("zip")),
            ttk.Button(btns, text=self.tr("choose_folder"), command=lambda: self.install("folder")),
        ]
        for b in self.setup_buttons:
            b.pack(side=side, padx=3)
        self.progress = ttk.Progressbar(self.setup_box, mode="determinate", maximum=100)
        self.progress_text = self.label(self.setup_box, "", bg=SAND, fg=MUTED)
        self.cancel_btn = ttk.Button(self.setup_box, text=self.tr("cancel"), command=self._cancel.set)
        self._setup_anchor = tk.Frame(h, bg=CARD, height=1)  # the models box goes just before the setup steps
        self._setup_anchor.pack(fill="x")

        # the whole setup on one card
        c = self.card(h, self.tr("quick_title"), self.tr("quick_note"))
        v = self.var
        s1 = self.step(c, 1, self.tr("step_platform"))
        names = [self.tr("plat_" + p) for p in ingest.PLATFORMS]
        plat = self.field(s1, self.tr("platform"), v("platform", tk.StringVar, ""), combo=names)
        plat.configure(state="readonly")  # type: ignore[call-arg]
        plat.bind("<<ComboboxSelected>>", lambda e: self.on_platform())
        self.platform_hint = self.small(s1, "")
        s2 = self.step(c, 2, self.tr("step_key"))
        url_entry = self.field(s2, self.tr("rtmp_url"), v("rtmp_url", tk.StringVar, ""))
        key_entry = self.field(s2, self.tr("stream_key"), v("key", tk.StringVar, ""), show="•")
        for ent, typed_server in ((url_entry, True), (key_entry, False)):
            ent.bind("<FocusOut>", lambda e, t=typed_server: self.tidy_target(t), add="+")
            ent.bind("<<Paste>>", lambda e, t=typed_server: self.after(50, lambda: self.tidy_target(t)), add="+")
        self.pack_row(ttk.Checkbutton(s2, text=self.tr("remember_key"), variable=v("remember", tk.BooleanVar, False)))
        if not self.ctrl.keys.available:
            self.small(s2, self.tr("key_no_store"))
        tf = self.buttons(s2, (self.tr("test_link"), self.on_test_link))
        self.test_result = self.label(tf, "", font=(FONT, 9, "bold"), wraplength=360)
        self.test_result.pack(side=side, padx=8)
        s3 = self.step(c, 3, self.tr("step_obs"))
        self.small(s3, self.tr("step_obs_note"))
        af = self.buttons(s3, (self.tr("obs_auto_btn"), self.on_obs_setup), (self.tr("obs_restore_btn"), self.on_obs_restore))
        self.obs_result = self.label(af, "", font=(FONT, 9, "bold"), wraplength=360)
        self.obs_result.pack(side=side, padx=8)
        self.small(s3, self.tr("obs_manual"))
        fr = tk.Frame(s3, bg=SAND, padx=10, pady=8)
        self.srt_hint = tk.Label(fr, text="", font=("Consolas", 11, "bold"), bg=SAND, fg=INK)
        ttk.Button(fr, text=self.tr("copy"), width=8, command=lambda: self.copy(self.obs_server())).pack(side=side)
        self.srt_hint.pack(side=side, padx=10)
        self.pack_row(fr, fill="x", pady=4)
        self.buttons(s3, (self.tr("obs_guide"), self.show_obs_guide))
        self.small(s3, self.tr("console_hint"))

        if release.SUPPORT_URL:  # launch period: free, with a friendly way to help
            sup = tk.Frame(h, bg=BLUSH, padx=16, pady=12, highlightthickness=1, highlightbackground="#f6b6d6")
            sup.pack(fill="x", pady=(0, 10))
            tk.Label(sup, text=self.tr("support_title"), font=(FONT, 12, "bold"), bg=BLUSH, fg=INK).pack(anchor=self.side)
            self.label(sup, self.tr("support_text"), bg=BLUSH, wraplength=520).pack(anchor=self.side, pady=(2, 8))
            kofi = Tile(
                sup, self.tr("support_button"), self.tr("support_sub"), DUSK, "#1a0716", lambda: webbrowser.open(release.SUPPORT_URL)
            )
            kofi._title.configure(font=(FONT, 12, "bold"))
            kofi.configure(padx=14, pady=6)
            kofi.pack(anchor=self.side)
        foot = tk.Frame(h, bg=CARD)
        foot.pack(fill="x", pady=(4, 0))
        self.report_btn = ttk.Button(foot, text=self.tr("open_report"), command=self.open_report)
        self.report_btn.pack(side=side)
        self.logbox = tk.Label(h, text="", bg=CARD, fg="#9a93a8", font=(FONT, 8), anchor="w", justify="left", wraplength=580)
        self.logbox.pack(fill="x", pady=(6, 0))
        v("input", tk.StringVar, "").trace_add("write", lambda *_: self.show_srt_hint())
        v("srt_port", tk.StringVar, "").trace_add("write", lambda *_: self.show_srt_hint())
        self.show_srt_hint()
        self.show_platform_hint()

    # --- settings: everything a normal stream never needs to touch, in plain boxes ------------------------
    def scroll_area(self, tab: tk.Frame) -> tk.Frame:
        """A frame inside ``tab`` that scrolls with the mouse wheel when it is taller than the window."""
        canvas = tk.Canvas(tab, bg=CARD, highlightthickness=0)
        bar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=CARD)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        bar.pack(side="left" if self.rtl else "right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def wheel(event: tk.Event) -> None:
            with contextlib.suppress(tk.TclError):  # a canvas of a window rebuilt since (language switch) is gone
                if canvas.winfo_exists() and str(self.nb.select()) == str(tab) and event.delta:
                    canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        self.bind_all("<MouseWheel>", wheel, add="+")
        return inner

    def build_settings(self) -> None:
        st = self.ctrl.settings
        s = self.settings_body = self.scroll_area(self.sets)
        v = self.var
        side = "right" if self.rtl else "left"

        def choice(parent: tk.Misc, text_key: str, value: str, name: str) -> None:
            self.pack_row(ttk.Radiobutton(parent, text=self.tr(text_key), value=value, variable=v(name, tk.StringVar, "")), pady=(8, 2))

        # where the picture comes from
        c = self.card(s, self.tr("sec_in"), self.tr("sec_in_note"))
        choice(c, "in_srt", cfg.INPUT_SRT, "input")
        self.field(c, self.tr("srt_port"), v("srt_port", tk.StringVar, ""), 8)
        choice(c, "in_camera", cfg.INPUT_CAMERA, "input")
        self.camera_box = self.field(c, self.tr("camera_name"), v("camera", tk.StringVar, ""), combo=[st.camera] if st.camera else [])
        self.sound_box = self.field(c, self.tr("camera_sound"), v("camera_sound", tk.StringVar, ""), combo=[""])
        self.buttons(c, (self.tr("refresh_devices"), self.refresh_devices))
        self.small(c, self.tr("camera_hint"))
        choice(c, "in_file", cfg.INPUT_FILE, "input")
        self.field(c, self.tr("input_file"), v("input_file", tk.StringVar, ""))
        self.buttons(c, (self.tr("pick_file"), self.pick_input_file))

        # where the protected picture goes
        c = self.card(s, self.tr("sec_out"), self.tr("sec_out_note"))
        choice(c, "out_rtmp", cfg.OUTPUT_RTMP, "output")
        choice(c, "out_record", cfg.OUTPUT_RECORD, "output")
        self.field(c, self.tr("record_dir"), v("record_dir", tk.StringVar, ""))
        self.buttons(c, (self.tr("pick_folder"), self.pick_record_dir))
        self.small(c, self.tr("record_hint", folder=self.ctrl.record_file().rsplit(os.sep, 1)[0]))
        choice(c, "out_window", cfg.OUTPUT_WINDOW, "output")
        self.small(c, self.tr("window_hint"))

        # picture quality: the same as OBS sends, nothing is made smaller or slower on the way
        c = self.card(s, self.tr("sec_quality"), self.tr("quality_hint"))
        self.pack_row(ttk.Checkbutton(c, text=self.tr("follow_obs"), variable=v("follow_obs", tk.BooleanVar, True)))
        res = self.field(c, self.tr("resolution"), v("resolution", tk.StringVar, ""), 10, combo=list(cfg.RESOLUTIONS))
        res.configure(state="readonly")  # type: ignore[call-arg]
        rate = self.field(c, self.tr("fps"), v("fps", tk.StringVar, ""), 10, combo=[str(f) for f in cfg.FPS_CHOICES])
        rate.configure(state="readonly")  # type: ignore[call-arg]
        delay = self.field(c, self.tr("delay"), v("delay", tk.StringVar, ""), 10, combo=[self.tr("delay_auto"), "2", "3", "4", "5"])
        delay.configure(state="readonly")  # type: ignore[call-arg]
        self.small(c, self.tr("delay_hint"))

        # how to cover, and the panic key
        c = self.card(s, self.tr("sec_protect"))
        choice(c, "cover_people", cfg.COVER_PEOPLE, "cover_mode")
        choice(c, "cover_screen", cfg.COVER_SCREEN, "cover_mode")
        choice(c, "sens_balanced", cfg.SENS_BALANCED, "sensitivity")
        choice(c, "sens_strict", cfg.SENS_STRICT, "sensitivity")
        pk = self.field(c, self.tr("panic_key"), v("panic_key", tk.StringVar, ""), 12, combo=hotkey.CHOICES)
        pk.configure(state="readonly")  # type: ignore[call-arg]
        pk.bind("<<ComboboxSelected>>", lambda e: self.on_panic_key())
        self.small(c, self.tr("panic_hint"))

        # a second platform through the protection
        c = self.card(s, self.tr("sec_multi"), self.tr("multi_note"))
        names2 = [self.tr("plat2_none")] + [self.tr("plat_" + p) for p in ingest.PLATFORMS]
        p2 = self.field(c, self.tr("platform"), v("platform2", tk.StringVar, ""), combo=names2)
        p2.configure(state="readonly")  # type: ignore[call-arg]
        self.field(c, self.tr("rtmp_url"), v("rtmp_url2", tk.StringVar, ""))
        self.field(c, self.tr("stream_key"), v("key2", tk.StringVar, ""), show="•")

        # the subscription (Dodo licence key)
        c = self.card(s, self.tr("sec_license"))
        self.license_label = self.label(c, "", font=(FONT, 10, "bold"), wraplength=520)
        self.pack_row(self.license_label, fill="x")
        self.field(c, self.tr("license_key"), self._vars["license_key"], show="•")
        lf = self.buttons(
            c, (self.tr("license_activate"), self.on_activate_license), (self.tr("license_deactivate"), self.on_deactivate_license)
        )
        buy = release.SUPPORT_URL or release.BUY_URL  # launch period: the Pro key comes with support on Ko-fi
        if buy:
            ttk.Button(lf, text=self.tr("license_buy"), command=lambda: webbrowser.open(buy)).pack(side=side, padx=4)
        self.show_license()

        # the logo on the cover (Pro)
        pro = plan.allows("custom_brand", self.ctrl.env)
        c = self.card(s, self.tr("sec_brand"), self.tr("brand_hint" if pro else "brand_pro_only"))
        fr = tk.Frame(c, bg=CARD)
        state = "normal" if pro else "disabled"  # the normal plan: our logo, not changeable
        ttk.Button(fr, text=self.tr("brand_pick"), command=self.pick_brand, state=state).pack(side=side, padx=4)
        ttk.Button(fr, text=self.tr("brand_remove"), command=lambda: self.set_brand(""), state=state).pack(side=side, padx=4)
        self.brand_label = self.label(fr, "", fg=MUTED, font=(FONT, 9))
        self.brand_label.pack(side=side, padx=8)
        self.pack_row(fr, fill="x")
        self.show_brand()

        # Arabic subtitles for English voice chat
        c = self.card(s, self.tr("sec_translate"))
        self.pack_row(
            ttk.Checkbutton(
                c, text=self.tr("translate_ar"), variable=v("translate", tk.BooleanVar, False), command=self.on_translate_toggle
            )
        )
        self.small(c, self.tr("translate_hint", mb=release.TRANSLATE_DOWNLOAD_MB))

        # files OBS reads (the panic switch script)
        c = self.card(s, self.tr("open_obs"))
        for key, path in (("obs_state", paths.obs_state_file(self.ctrl.env)), ("obs_script", paths.obs_script())):
            fr = tk.Frame(c, bg=CARD)
            ttk.Button(fr, text=self.tr("copy"), width=8, command=lambda p=str(path): self.copy(p)).pack(
                side="right" if self.rtl else "left", padx=4
            )
            tk.Label(fr, text=self.tr(key, path=path), bg=CARD, fg=MUTED, font=(FONT, 8), anchor="w", justify="left", wraplength=470).pack(
                side="left" if self.rtl else "right", fill="x", expand=True
            )
            self.pack_row(fr, fill="x")

    # --- the links: OBS, platform, devices, recordings ---------------------------------------------------
    def obs_server(self) -> str:
        port = str(self._vars["srt_port"].get()).strip() or "9000"
        return f"srt://127.0.0.1:{port}"

    def platform_value(self) -> str:
        """The platform id of the name shown in the list."""
        shown = str(self._vars["platform"].get()).replace(RLE, "").replace(PDF, "")
        names = {i18n.t(lang, "plat_" + p): p for p in ingest.PLATFORMS for lang in i18n.LANGS}  # either language: a rebuild
        return names.get(shown, "custom")

    def on_platform(self) -> None:
        """A platform picked: the pick wins. Its own server goes into the box; a server of another platform (or one
        of ours) is replaced, so a server left from before can never send the stream to the wrong platform."""
        plat, url = self.platform_value(), str(self._vars["rtmp_url"].get()).strip()
        known = {ingest.normalise(u) for u in ingest.PRESETS.values() if u}
        other = ingest.guess_platform(url) not in ("custom", plat)
        if not url or ingest.normalise(url) in known or other:
            self._vars["rtmp_url"].set(ingest.PRESETS.get(plat, ""))
        self.ctrl.settings.platform = plat
        self.show_platform_hint()

    def show_platform_hint(self) -> None:
        with contextlib.suppress(tk.TclError, AttributeError):
            self.platform_hint.configure(text=self.tr("plat_hint_" + self.platform_value()))

    def tidy_target(self, typed_server: bool = True) -> None:
        """After a paste or leaving a box: a whole address is split into server and key, spaces go, Kick's server
        gets its ``/app``. The platform follows the address only when the streamer gave the address (server box,
        or a whole address pasted in the key box), never from a server left in the box (a key paste)."""
        url, key = str(self._vars["rtmp_url"].get()), str(self._vars["key"].get())
        if not url.strip() and not key.strip():
            return
        self.ctrl.settings.platform = self.platform_value()
        server, clean_key = self.ctrl.take_pasted(url, key)
        if server != url:
            self._vars["rtmp_url"].set(server)
        if clean_key != key:
            self._vars["key"].set(clean_key)
        guess = ingest.guess_platform(server)
        if (typed_server or server != url) and guess != "custom" and guess != self.platform_value():
            self._vars["platform"].set(self.tr("plat_" + guess))
        self.show_platform_hint()

    def show_obs_guide(self) -> None:
        messagebox.showinfo(self.tr("obs_guide"), self.tr("obs_guide_text", server=self.obs_server()))

    def refresh_devices(self) -> None:
        def work() -> None:
            found = self.ctrl.devices()

            def show() -> None:
                self.camera_box.configure(values=found["video"])  # type: ignore[call-arg]
                self.sound_box.configure(values=["", *found["audio"]])  # type: ignore[call-arg]
                if not found["video"]:
                    messagebox.showinfo(PRODUCT_NAME, self.tr("no_devices"))

            self.post(show)

        threading.Thread(target=work, name="devices", daemon=True).start()

    def pick_input_file(self) -> None:
        path = filedialog.askopenfilename(
            title=self.tr("pick_file"), filetypes=[(self.tr("videos"), "*.mp4 *.mkv *.mov *.avi *.flv *.ts *.webm"), ("*", "*.*")]
        )
        if path:
            self._vars["input_file"].set(path)
            self._vars["input"].set(cfg.INPUT_FILE)
            self._vars["output"].set(cfg.OUTPUT_RECORD)

    def pick_record_dir(self) -> None:
        path = filedialog.askdirectory(title=self.tr("pick_folder"))
        if path:
            self._vars["record_dir"].set(path)

    def on_test_link(self) -> None:
        self.collect()
        self.tidy_target(typed_server=False)
        self.collect()
        key = str(self._vars["key"].get()).strip()
        problems = self.ctrl.target(key).problems
        if problems:
            self.test_result.configure(text=self.tr(problems[0]), fg="#b3261e")
            return
        if self.ctrl.engine.running or self._busy.is_set():
            self.test_result.configure(text=self.tr("test_busy"), fg="#b3261e")
            return
        if not messagebox.askyesno(PRODUCT_NAME, self.tr("test_warn")):
            return
        self.test_result.configure(text=self.tr("test_running"), fg="#44505e")

        def work() -> None:
            why = self.ctrl.test_connection(key)

            def show() -> None:
                if why:
                    self.test_result.configure(text=self.tr("test_fail", why=self.tr("why_" + why)), fg="#b3261e")
                else:
                    self.test_result.configure(text=self.tr("test_ok"), fg="#1b7f3b")

            self.post(show)

        threading.Thread(target=work, name="link-test", daemon=True).start()

    def show_license(self) -> None:
        from streamsafe.app import license as lic

        held = lic.load(self.ctrl.env)
        st = self.ctrl.license_status
        sp = passkey.load(self.ctrl.env)
        if held is None and sp is not None:  # a supporter's pass: Pro until its last day
            why = passkey.why_off(self.ctrl.env)
            if not why:
                text, color = self.tr("pass_on", date=sp.expires.isoformat(), who=sp.who or "-"), "#1b7f3b"
            else:
                text, color = self.pass_reason(why, sp.expires.isoformat()), "#b26a00"
        elif held is not None:
            plan_name = self.tr("plan_" + held.plan)
            text = self.tr("license_on", plan=plan_name, key=lic.masked(held.key)) + (self.tr("license_grace") if st.grace else "")
            color = "#1b7f3b"
        elif release.LICENSE_REQUIRED:
            text, color = self.tr("license_needed"), "#b3261e"
        else:
            text, color = self.tr("license_beta"), "#44505e"
        with contextlib.suppress(tk.TclError, AttributeError):
            self.license_label.configure(text=text, fg=color)

    def pass_reason(self, why: str, date: str) -> str:
        """Why a supporter's key gives no Pro, in words."""
        return {
            "invalid": self.tr("pass_invalid"),
            "clock": self.tr("pass_clock"),
            "expired": self.tr("pass_ended", date=date) if date else self.tr("pass_expired"),
        }.get(why, self.tr("pass_invalid"))

    def sync_pass_clock(self) -> None:
        """Off the window's thread: a supporter's key follows the internet's date, not only the PC clock."""
        with contextlib.suppress(Exception):
            passkey.sync_clock(self.ctrl.env)
            self.post(self.show_license)

    def on_activate_license(self) -> None:
        key = str(self._vars["license_key"].get())

        def work() -> str:
            if passkey.is_pass(key):  # a supporter's pass: checked here (and the internet's date)
                why = passkey.activate(key, self.ctrl.env)
                return self.pass_reason(why, "") if why else ""
            st = self.ctrl.activate_license(key)
            return "" if st.ok else self.tr("why_license_" + (st.reason or "invalid"))

        def done(err: str) -> None:
            self._vars["license_key"].set("")
            if err:
                messagebox.showwarning(PRODUCT_NAME, err)
            self.build()  # the plan may have changed what the window offers (the logo)
            self.nb.select(self.sets)

        self.job("license", work, done)

    def on_deactivate_license(self) -> None:
        if not messagebox.askyesno(PRODUCT_NAME, self.tr("license_deactivate_ask")):
            return

        def work() -> str:
            from streamsafe.app import license as lic

            passkey.forget(self.ctrl.env)
            if lic.load(self.ctrl.env) is None:
                return ""
            return "" if self.ctrl.deactivate_license() else self.tr("license_deactivate_offline")

        def done(err: str) -> None:
            if err:
                messagebox.showwarning(PRODUCT_NAME, err)
            self.build()
            self.nb.select(self.sets)

        self.job("license", work, done)

    def show_brand(self) -> None:
        path = self.ctrl.settings.brand_image if plan.allows("custom_brand", self.ctrl.env) else ""
        text = self.tr("brand_set", name=os.path.basename(path)) if path else self.tr("brand_ours")
        self.brand_label.configure(text=text)

    def pick_brand(self) -> None:
        source = filedialog.askopenfilename(title=self.tr("brand_pick_title"), filetypes=[("PNG / JPG", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if source:
            self.set_brand(source)

    def set_brand(self, source: str) -> None:
        problem = self.ctrl.set_brand_image(source)
        if problem:
            messagebox.showwarning(PRODUCT_NAME, self.tr(problem))
        self.show_brand()

    def show_srt_hint(self) -> None:
        with contextlib.suppress(tk.TclError, AttributeError):  # a trace may fire while the window is being rebuilt
            self.srt_hint.configure(text=self.obs_server())

    def collect(self) -> None:
        """Window fields -> settings (values that are not valid stay as typed so that ``problems`` reports them)."""
        st, v = self.ctrl.settings, self._vars
        st.lang = self.lang
        st.input, st.output = str(v["input"].get()), str(v["output"].get())
        st.camera, st.rtmp_url = str(v["camera"].get()).strip(), str(v["rtmp_url"].get()).strip()
        st.camera_sound, st.input_file = str(v["camera_sound"].get()).strip(), str(v["input_file"].get()).strip()
        st.record_dir = str(v["record_dir"].get()).strip()
        st.platform = self.platform_value()
        shown2 = str(v["platform2"].get()).replace(RLE, "").replace(PDF, "")
        names2 = {i18n.t(lang, "plat_" + p): p for p in ingest.PLATFORMS for lang in i18n.LANGS}
        st.platform2 = names2.get(shown2, "")
        st.rtmp_url2 = str(v["rtmp_url2"].get()).strip()
        self.ctrl.key2 = str(v["key2"].get()).strip()
        st.remember_key = bool(v["remember"].get())
        st.translate_ar = bool(v["translate"].get())
        st.follow_obs = bool(v["follow_obs"].get())
        sens = str(v["sensitivity"].get())
        st.sensitivity = sens if sens in cfg.THR_SCALE else cfg.SENS_BALANCED
        mode = str(v["cover_mode"].get())
        st.cover_mode = mode if mode in (cfg.COVER_PEOPLE, cfg.COVER_SCREEN) else cfg.COVER_PEOPLE
        st.panic_button = str(v["panic_key"].get()) if hotkey.parse(str(v["panic_key"].get())) else st.panic_button
        try:
            st.srt_port = int(str(v["srt_port"].get()).strip())
        except ValueError:
            st.srt_port = 0
        shown = str(v["delay"].get()).strip().replace(",", ".")
        try:
            st.delay = 0.0 if shown in (self.tr("delay_auto"), i18n.t(self.lang, "delay_auto"), "") else float(shown)
        except ValueError:
            st.delay = 0.0
        st.resolution = str(v["resolution"].get()) if str(v["resolution"].get()) in cfg.RESOLUTIONS else "1080p"
        st.fps = int(v["fps"].get()) if str(v["fps"].get()) in ("30", "60") else 60

    # --- check tab ------------------------------------------------------------------------------------------
    def build_check(self) -> None:
        c = self.chk
        top = tk.Frame(c, bg="white")
        top.pack(fill="x")
        self.verdict = self.label(top, "", font=(FONT, 11, "bold"), wraplength=380)
        self.recheck_btn = ttk.Button(top, text=self.tr("recheck"), command=self.recheck_by_user)
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
            self.setup_box.pack(fill="x", pady=8, before=self._setup_anchor)
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

    def recheck_by_user(self) -> None:
        """The streamer pressed "check again": a past "not now" to the NVIDIA fast mode is asked again."""
        self.ctrl.settings.nvidia_declined = False
        self.refresh_check()

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

    def show_link(self) -> None:
        """The two links in plain words (from the engine's STATUS lines), redrawn only when they change."""
        lk, st = self.ctrl.link, self.ctrl.settings
        running = self.ctrl.engine.running
        shown = (
            running,
            lk.input,
            lk.platform,
            lk.sent,
            lk.reason,
            lk.retry_s,
            lk.platform2,
            self.ctrl.preview,
            lk.record_file,
            lk.recorded,
            st.srt_port,
            self.ctrl.port_moved_from,
            self.lang,
        )
        if shown == self._link_shown:
            return
        self._link_shown = shown
        if not running and not lk.stopped:
            self.link_box.pack_forget()
            return
        green, amber, red, grey = "#1b7f3b", "#b26a00", "#b3261e", "#44505e"
        server = self.ctrl.obs_server()
        why = self.tr("why_" + lk.reason) if lk.reason else ""
        if st.input == cfg.INPUT_SRT:
            text, color = {
                "connected": (self.tr("lk_obs_ok"), green),
                "disconnected": (self.tr("lk_obs_lost", server=server), amber),
                "failed": (self.tr("lk_obs_failed", why=why), red),
            }.get(lk.input, (self.tr("lk_obs_wait", server=server), amber))
            if self.ctrl.port_moved_from and lk.input != "connected":
                text = self.tr("port_moved", old=self.ctrl.port_moved_from, new=st.srt_port, server=server) + "\n" + text
        else:
            text, color = (self.tr("lk_in_ok"), green) if lk.input == "connected" or running else (self.tr("lk_in_wait"), grey)
            if lk.input == "failed":
                text, color = self.tr("lk_in_failed", why=why), red
        warn = self.format_warning()
        if warn and lk.input == "connected":
            text, color = text + "\n" + warn, amber
        self.lk_in.configure(text=text, fg=color)
        output = cfg.OUTPUT_WINDOW if self.ctrl.preview else st.output
        if output == cfg.OUTPUT_RTMP:
            text, color = {
                "connecting": (self.tr("lk_plat_connecting"), amber),
                "connected": (self.tr("lk_plat_ok"), green),
                "lost": (self.tr("lk_plat_lost", why=why, s=f"{lk.retry_s:g}"), amber),
                "failed": (self.tr("lk_plat_failed", why=why), red),
            }.get(lk.platform, (self.tr("lk_plat_wait"), grey))
            if st.platform2:
                state = {"connected": "✓", "lost": "…", "failed": "✗", "connecting": "…"}.get(lk.platform2, "…")
                text += "\n" + self.tr("link_out2", state=self.tr("plat_" + st.platform2) + " " + state)
        elif output == cfg.OUTPUT_RECORD:
            text, color = (
                (self.tr("lk_rec_done", file=lk.record_file), green)
                if lk.recorded
                else ((self.tr("lk_rec", file=lk.record_file), green) if lk.record_file else (self.tr("lk_rec_wait"), grey))
            )
        else:
            text, color = self.tr("lk_window"), grey
        self.lk_out.configure(text=text, fg=color)
        if st.input == cfg.INPUT_SRT and lk.input != "connected" and running:
            self.lk_copy.pack(anchor=self.side, pady=(4, 0))
        else:
            self.lk_copy.pack_forget()
        self.link_box.pack(fill="x", pady=6, after=self.message.master)

    def format_warning(self) -> str:
        """When OBS sends more than the protection keeps (a bigger picture, more frames, or too low a bitrate to
        start from), say so: the quality is decided by the smallest link of the chain."""
        sent = self.ctrl.link.sent
        if not sent:
            return ""
        _w, h, fps, kbps = sent
        st = self.ctrl.settings
        _, our_h = cfg.RESOLUTIONS.get(st.resolution, (1920, 1080))
        if h > our_h + 8 or fps > st.fps + 1:
            return self.tr("sent_bigger", sent=f"{h}p{round(fps) or ''}", ours=f"{our_h}p{st.fps}")
        if kbps and kbps < 12000:
            return self.tr("sent_low_bitrate", kbps=kbps)
        return ""

    def update_controls(self) -> None:
        self.show_link()
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

    def on_start(self, preview: bool = False) -> None:
        self.collect()
        key = str(self._vars["key"].get()).strip()
        self.message.configure(text="", fg="#b3261e")
        self.pre_note.configure(text="\n".join(self.tr(w) for w in preflight.warnings(self.ctrl.env)))
        self._announced_exit = False
        why = self.ctrl.blockers(key, replace(self.ctrl.settings, output=cfg.OUTPUT_WINDOW) if preview else None)
        if why:
            self.message.configure(text=self.tr("not_ready", why=self.tr(why[0])))
            return
        slow = self.ctrl.has_nvidia_gpu() and not self.ctrl.use_nvidia and bool(self.ctrl.check_result.verdict) and not self._busy.is_set()
        # an NVIDIA card on the slow path: the field report's slates every few seconds. Offer the fast mode first.
        if slow and messagebox.askyesno(PRODUCT_NAME, self.tr("nvidia_needed", gb=release.NVIDIA_DOWNLOAD_GB)):
            self.enable_nvidia()
            return
        if not self.ctrl.remember(key):
            self._vars["remember"].set(False)
        self.ctrl.remember2(self.ctrl.key2)
        engine.follow_obs(self.ctrl.settings, obs_setup.obs_video())  # the streamer's own OBS quality, nothing to set
        self.ctrl.settings.obs_kbps = obs_setup.obs_kbps()
        self._vars["resolution"].set(self.ctrl.settings.resolution)
        self._vars["fps"].set(str(self.ctrl.settings.fps))
        self.ctrl.save_settings()
        try:
            self.ctrl.start(key, preview=preview)
        except (RuntimeError, OSError) as exc:
            why = self.tr("sac_blocked") if preflight.blocked_by_windows(exc) else self.tr(str(exc)) if str(exc) in i18n.TEXT else str(exc)
            self.message.configure(text=self.tr("not_ready", why=why))
        else:
            if preview:
                self.message.configure(text=self.tr("preview_on"), fg="#44505e")
            if self.ctrl.settings.input == cfg.INPUT_SRT:  # the address read here: window values stay on this thread
                threading.Thread(target=self.bring_obs, args=(self.obs_server(),), name="obs", daemon=True).start()
        self._last_status = ""
        self.show_status(force=True)

    # --- OBS set up by itself -----------------------------------------------------------------------------
    def obs_text(self, why: str) -> str:
        return {
            "": self.tr("obs_auto_done"),
            "running": self.tr("obs_auto_running"),
            "no_obs": self.tr("obs_auto_no_obs"),
        }.get(why, self.tr("obs_auto_error"))

    def on_obs_setup(self) -> None:
        """Step 3's button: point OBS at us (OBS closed), once."""
        self.collect()
        why = obs_setup.apply(self.obs_server())
        self.obs_result.configure(text=self.obs_text(why), fg="#1b7f3b" if not why else "#b3261e")

    def bring_obs(self, server: str) -> None:
        """After Start (worker thread, no window access but ``post``): OBS closed -> set it up if needed and open it
        streaming to us by itself; OBS open -> say what to press, or that it must be closed once to be set up."""
        if obs_setup.obs_running():
            text = self.tr("obs_press_start") if obs_setup.is_set(server) else self.tr("obs_auto_running")
            self.post(lambda: self.message.configure(text=text, fg="#44505e"))
            return
        why = "" if obs_setup.is_set(server) else obs_setup.apply(server)
        exe = obs_setup.find_obs()
        if why or exe is None:
            text = self.obs_text(why or "no_obs")
            self.post(lambda: self.message.configure(text=text, fg="#b3261e"))
            return
        if obs_setup.launch(exe):
            self.post(lambda: self.message.configure(text=self.tr("obs_opening"), fg="#44505e"))

    # --- the panic key and OBS put back ---------------------------------------------------------------------------
    def arm_panic_key(self) -> None:
        """(Re)registers the streamer's panic key; says so when another program holds it."""
        if self._hotkey is not None:
            self._hotkey.stop()
            self._hotkey = None
        name = self.ctrl.settings.panic_button
        if not name:
            return
        hk = hotkey.Hotkey(name, lambda: self.post(self.on_panic))
        if hk.start():
            self._hotkey = hk
        elif sys.platform == "win32":
            self.pre_note.configure(text=self.tr("panic_taken", hk=name))

    def on_panic_key(self) -> None:
        self.collect()
        self.ctrl.save_settings()
        self.arm_panic_key()

    def on_panic(self) -> None:
        on = not self.ctrl.panic
        self.ctrl.set_panic(on)
        text = self.tr("panic_on", hk=self.ctrl.settings.panic_button) if on else self.tr("panic_off")
        self.message.configure(text=text, fg="#b3261e" if on else "#1b7f3b")

    def on_obs_restore(self) -> None:
        why = obs_setup.restore()
        key = {"": "obs_restore_done", "no_backup": "obs_restore_none", "running": "obs_restore_running"}.get(why, "obs_auto_error")
        self.obs_result.configure(text=self.tr(key), fg="#1b7f3b" if not why else "#b3261e")

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

    def on_translate_toggle(self) -> None:
        """Switched on for the first time: download the translation files (the stream never waits for them)."""
        var = self._vars["translate"]
        if not var.get() or self.ctrl.translate_problem() == "":
            return  # switched off, or the files are already here
        if self._busy.is_set():
            var.set(False)  # busy with another download: ask again later
            return

        def work() -> str:
            why = self.ctrl.install_translate(lambda w, d, t: self.on_progress("downloading", d, t), self._cancel)
            return self.tr("translate_failed", why=why) if why else ""

        def done(err: str) -> None:
            if err:
                var.set(False)  # no files: translation stays off
            else:
                self.message.configure(text=self.tr("translate_ready"), fg="#1b7f3b")
            self.collect()
            self.ctrl.save_settings()

        self.job("translate", work, done)

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
        if app_new and self.ctrl.update and self.ctrl.update.app_sha256:
            self.app_update_btn.pack(side="left" if self.rtl else "right", padx=6)
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

    def apply_app_update(self) -> None:
        """One click: download the new version, check it, install it over this one, open again. Never while live."""
        if self.ctrl.engine.running:
            self.message.configure(text=self.tr("update_busy"))
            return
        got: list[Path] = []

        def work() -> str:
            try:
                got.append(self.ctrl.download_app_update(lambda w, d, t: self.on_progress(w, d, t), self._cancel))
            except Exception as exc:
                return self.tr("update_failed", why=str(exc)) if not self._cancel.is_set() else self.tr("install_cancelled")
            return ""

        def done(err: str) -> None:
            if err or not got:
                self.message.configure(text=err, fg="#b3261e")
                return
            try:
                self.ctrl.run_installer(got[0])
            except OSError as exc:
                why = self.tr("sac_blocked") if preflight.blocked_by_windows(exc) else str(exc)
                self.message.configure(text=self.tr("update_failed", why=why), fg="#b3261e")
                return
            if self._hotkey is not None:
                self._hotkey.stop()
            self.destroy()  # the installer closes nothing of ours that is still open, then opens the new version

        self.job("update", work, done)

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
        if self._hotkey is not None:
            self._hotkey.stop()
        self.ctrl.set_panic(False)
        self._cancel.set()
        self.destroy()


def main(argv: list[str] | None = None) -> int:
    if not single_instance():
        return 0
    app = App()
    app.mainloop()
    return 0
