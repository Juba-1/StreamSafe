"""The streamer's list of pictures to always cover, kept per game (data folder/pictures/).

    library.json  [{"id", "game", "name", "rect": [x0, y0, x1, y1], "full": bool, "on": bool}, ...]
    <id>.png      the picture itself, cut from the window's preview at that place

A picture is taken from a frozen copy of the preview (``freeze``) so what the streamer marked is exactly what is
saved. The engine gets the pictures that are on through the live control file (runtime/pictures.py).
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import shutil
from pathlib import Path

import cv2

from streamsafe.runtime.control import rects

FROZEN = "_frozen.png"
MAX_PICTURES = 60


class Library:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.items: list[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads((self.folder / "library.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = []
        self.items = [
            i for i in data if isinstance(i, dict) and isinstance(i.get("id"), str) and (self.folder / f"{i['id']}.png").is_file()
        ]

    def _save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.folder / "library.json.tmp"
        tmp.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.folder / "library.json")

    def file(self, pid: str) -> Path | None:
        """The picture's file (or the frozen preview), only for known names."""
        if pid == "_frozen":
            f = self.folder / FROZEN
        elif any(i["id"] == pid for i in self.items):
            f = self.folder / f"{pid}.png"
        else:
            return None
        return f if f.is_file() else None

    def freeze(self, preview: Path) -> bool:
        """Keep the current preview still while the streamer marks a picture on it."""
        if not preview.is_file():
            return False
        self.folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(preview, self.folder / FROZEN)
        return True

    def add(self, game: str, name: str, rect: list, full: bool) -> str:
        """Cut the marked part out of the frozen preview and keep it. Returns "" or the reason it was not kept."""
        r = rects([rect])
        if not r:
            return "pic_bad_rect"
        if len(self.items) >= MAX_PICTURES:
            return "pic_too_many"
        img = cv2.imread(str(self.folder / FROZEN))
        if img is None:
            return "pic_no_preview"
        h, w = img.shape[:2]
        x0, y0, x1, y1 = r[0]
        part = img[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)]
        if part.shape[0] < 8 or part.shape[1] < 8:
            return "pic_bad_rect"
        pid = secrets.token_hex(6)
        if not cv2.imwrite(str(self.folder / f"{pid}.png"), part):
            return "pic_no_preview"
        game = (game.strip() or "—")[:40]
        self.items.append(
            {"id": pid, "game": game, "name": (name.strip() or game)[:40], "rect": list(r[0]), "full": bool(full), "on": True}
        )
        self._save()
        return ""

    def remove(self, pid: str) -> None:
        self.items = [i for i in self.items if i["id"] != pid]
        with contextlib.suppress(OSError):
            (self.folder / f"{pid}.png").unlink()
        self._save()

    def set(self, pid: str, **values: object) -> None:
        for i in self.items:
            if i["id"] == pid:
                for k in ("on", "full"):
                    if k in values:
                        i[k] = bool(values[k])
        self._save()

    def set_game(self, game: str, on: bool) -> None:
        """Turn all pictures of one game on or off (the streamer switches games)."""
        for i in self.items:
            if i["game"] == game:
                i["on"] = on
        self._save()

    def active(self) -> list[dict]:
        """For the live control file."""
        return [{"rect": i["rect"], "file": str(self.folder / f"{i['id']}.png"), "full": i["full"]} for i in self.items if i.get("on")]
