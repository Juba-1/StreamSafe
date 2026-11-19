"""The streamer's panic key: one key, anywhere in Windows (also inside a full-screen game), covers the whole screen;
pressed again it hands back to the automatic cover. Windows' own RegisterHotKey, no keyboard hook.

``Hotkey("Ctrl+F9", on_press).start()``; ``stop()`` frees the key. Not on Windows, or a key another program holds:
``start()`` returns False and the window says so (the automatic cover is not affected).
"""

from __future__ import annotations

import contextlib
import sys
import threading
from collections.abc import Callable

WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
MODS = {"ctrl": 0x0002, "alt": 0x0001, "shift": 0x0004}
MOD_NOREPEAT = 0x4000
KEYS: dict[str, int] = {f"F{n}": 0x6F + n for n in range(1, 13)}
KEYS.update({"Pause": 0x13, "ScrollLock": 0x91, "Insert": 0x2D, "Home": 0x24, "End": 0x23, "PageUp": 0x21, "PageDown": 0x22})
KEYS.update({f"Num{n}": 0x60 + n for n in range(10)})
CHOICES = ["F9", "F10", "F11", "F12", "F8", "F7", "Pause", "ScrollLock", "Insert", "Home", "End", "PageUp", "PageDown"]
CHOICES += ["Ctrl+F9", "Ctrl+F10", "Alt+F9", "Shift+F9", *[f"Num{n}" for n in range(10)]]


def parse(text: str) -> tuple[int, int] | None:
    """'Ctrl+F9' -> (modifiers, virtual key); None if it is not a key we offer."""
    parts = [p.strip() for p in text.split("+") if p.strip()]
    if not parts:
        return None
    key = next((k for k in KEYS if k.lower() == parts[-1].lower()), None)
    if key is None:
        return None
    mods = 0
    for p in parts[:-1]:
        if p.lower() not in MODS:
            return None
        mods |= MODS[p.lower()]
    return mods, KEYS[key]


class Hotkey:
    def __init__(self, text: str, on_press: Callable[[], None]) -> None:
        self.text, self.on_press = text, on_press
        self._thread: threading.Thread | None = None
        self._tid = 0
        self._ok = threading.Event()
        self._done = threading.Event()

    def start(self) -> bool:
        key = parse(self.text)
        if key is None or sys.platform != "win32":
            return False
        self._thread = threading.Thread(target=self._loop, args=key, name="panic-key", daemon=True)
        self._thread.start()
        self._done.wait(2.0)
        return self._ok.is_set()

    def _loop(self, mods: int, vk: int) -> None:
        import ctypes
        from ctypes import wintypes

        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32  # type: ignore[attr-defined]
        self._tid = kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, 1, mods | MOD_NOREPEAT, vk):  # another program already holds this key
            self._done.set()
            return
        self._ok.set()
        self._done.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    with contextlib.suppress(Exception):  # a broken callback must not end the key
                        self.on_press()
        finally:
            user32.UnregisterHotKey(None, 1)

    def stop(self) -> None:
        if self._thread is not None and self._tid and sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)  # type: ignore[attr-defined]
            self._thread.join(1.0)
        self._thread = None
