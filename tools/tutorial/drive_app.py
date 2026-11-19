"""Tutorial recording on a Windows runner: the real StreamSafe window, driven like a streamer would with the real
mouse (pyautogui), while FFmpeg records the screen. Writes actions.json (time + screen box of every step) for the
edit (captions, zoom, the key blurred).

The parts a runner cannot do are stood in for, and the edit says so by cutting away:
- no GPU: the engine is the models-free self test (every frame covered), sending to a local stand-in for the
  platform; the real protection and the real platform are shown from the GPU pod's live run;
- the 4 GB model download is skipped (the window is shown as after it).
The key typed is a placeholder, shown masked, and blurred in the edit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pyautogui

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tutorial_out")
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("STREAMSAFE_DATA_DIR", str(OUT / "data"))
T0 = time.time()
ACTIONS: list[dict] = []
pyautogui.FAILSAFE = False

from streamsafe.app import controller, engine, ui  # noqa: E402

LOCAL_PLATFORM = "rtmp://127.0.0.1:19350/app/demo"


def selftest_args(s, reports, record, fast=False):  # the runner has no GPU: the models-free live path
    return ["selftest-stream", "--input", f"srt://127.0.0.1:{s.srt_port}?mode=listener", "--output", LOCAL_PLATFORM,
            "--delay", "2", "--size", "1280x720", "--report-dir", str(reports)]  # fmt: skip


engine.run_args = selftest_args
controller.Controller.models_problem = lambda self: ""
controller.Controller.heads_problem = lambda self: ""
_blockers = controller.Controller.blockers
controller.Controller.blockers = lambda self, key="": [b for b in _blockers(self, key) if b != "why_check"]  # no GPU here

app = ui.App()
app.lang = "ar"
app.build()
app.geometry("700x1000+1180+20")


def mark(name: str, widget=None, **extra) -> None:
    box = None
    if widget is not None:
        app.update()
        box = [widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_width(), widget.winfo_height()]
    ACTIONS.append({"t": round(time.time() - T0, 2), "step": name, "box": box, **extra})
    (OUT / "actions.json").write_text(json.dumps(ACTIONS, ensure_ascii=False, indent=1), encoding="utf-8")
    print("STEP", name, box, flush=True)


def glide(widget, click: bool = True, dur: float = 0.9) -> None:
    app.update()
    x = widget.winfo_rootx() + widget.winfo_width() // 2
    y = widget.winfo_rooty() + widget.winfo_height() // 2
    pyautogui.moveTo(x, y, duration=dur, tween=pyautogui.easeInOutQuad)
    if click:
        time.sleep(0.25)
        pyautogui.click()
    pump(0.6)


def pump(seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.03)


def find(widget_type, text: str):
    """The first widget of a type whose text contains ``text``."""
    stack = [app]
    while stack:
        w = stack.pop()
        try:
            if isinstance(w, widget_type) and text in str(w.cget("text")):
                return w
        except Exception:
            pass
        stack.extend(w.winfo_children())
    raise LookupError(text)


def scroll_home_to(widget) -> None:
    """Scroll the Home tab so ``widget`` is in view (the wheel, like a person)."""
    app.update()
    for _ in range(30):
        y = widget.winfo_rooty()
        if 120 < y < app.winfo_rooty() + app.winfo_height() - 160:
            return
        pyautogui.moveTo(app.winfo_rootx() + 300, app.winfo_rooty() + 500, duration=0.2)
        pyautogui.scroll(-120 if y > 500 else 120)
        pump(0.15)


def run() -> None:
    from tkinter import ttk

    pump(2.0)
    mark("app_open", app)
    pump(2.5)
    # step 1: the platform
    platform_box = find_combo()
    scroll_home_to(platform_box)
    mark("step1_platform", platform_box)
    glide(platform_box)
    pump(0.8)
    app._vars["platform"].set(app.tr("plat_kick"))
    app.on_platform()
    pyautogui.press("escape")
    pump(1.2)
    mark("step1_done", platform_box)
    # step 2: the key (a placeholder, masked)
    key_entry = find_key_entry()
    mark("step2_key", key_entry)
    glide(key_entry)
    pyautogui.typewrite("sk_placeholder_demo_key_0000", interval=0.03)
    pump(1.0)
    mark("step2_done", key_entry)
    # step 3: OBS set up by the program
    obs_btn = find(ttk.Button, app.tr("obs_auto_btn"))
    scroll_home_to(obs_btn)
    mark("step3_obs", obs_btn)
    glide(obs_btn)
    pump(2.0)
    mark("step3_done", app.obs_result)
    # start
    scroll_home_to(app.start_btn)
    pump(0.5)
    mark("start", app.start_btn)
    glide(app.start_btn)
    for _ in range(90):  # OBS opens and connects by itself
        pump(1.0)
        if any('"obs_connected"' in line for line in list(app.ctrl._engine_lines)):
            break
    mark("running", app.badge)
    pump(6.0)
    # stop: OBS first, then the program
    subprocess.run(["taskkill", "/IM", "obs64.exe"], capture_output=True)
    pump(3.0)
    mark("stop", app.stop_btn)
    glide(app.stop_btn)
    pump(4.0)
    mark("end", app)


def find_combo():
    from tkinter import ttk

    stack = [app.home]
    while stack:
        w = stack.pop()
        if isinstance(w, ttk.Combobox) and str(w.cget("textvariable")) == str(app._vars["platform"]):
            return w
        stack.extend(w.winfo_children())
    raise LookupError("platform combo")


def find_key_entry():
    import tkinter as tk

    stack = [app.home]
    while stack:
        w = stack.pop()
        if isinstance(w, tk.Entry) and str(w.cget("textvariable")) == str(app._vars["key"]):
            return w
        stack.extend(w.winfo_children())
    raise LookupError("key entry")


try:
    run()
finally:
    mark("done")
    app.ctrl.stop()  # the engine never outlives the recording
    app.destroy()
