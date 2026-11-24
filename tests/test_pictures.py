"""Pictures always covered: the library (per game) and the watch that covers them when they show."""

import json

import cv2
import numpy as np

from streamsafe.app import controller
from streamsafe.app.pictures import Library
from streamsafe.runtime.control import parse
from streamsafe.runtime.pictures import FULL, PictureWatch


def scene(with_map: bool) -> np.ndarray:
    rng = np.random.default_rng(1)
    f = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)  # a busy game picture
    if with_map:
        f[100:500, 1300:1800] = 40
        cv2.putText(f, "MAP", (1400, 330), cv2.FONT_HERSHEY_SIMPLEX, 4, (220, 200, 90), 12)
        cv2.circle(f, (1550, 420), 50, (30, 200, 240), -1)
    return f


def test_library_and_watch(tmp_path):
    lib = Library(tmp_path / "pictures")
    preview = tmp_path / "preview.png"
    cv2.imwrite(str(preview), cv2.resize(scene(True), (640, 360), interpolation=cv2.INTER_AREA))
    assert lib.freeze(preview)
    rect = [1300 / 1920, 100 / 1080, 1800 / 1920, 500 / 1080]
    assert lib.add("GTA 6", "Map", rect, False) == ""
    assert lib.add("GTA 6", "tiny", [0.1, 0.1, 0.101, 0.101], False) == "pic_bad_rect"
    live = parse({"pictures": lib.active()})
    assert len(live.pictures) == 1

    t = [0.0]
    watch = PictureWatch(clock=lambda: t[0])
    assert watch(scene(False), live.pictures) == []  # not showing: nothing covered
    assert watch(scene(True), live.pictures) == [tuple(rect)]  # showing (full resolution, other noise): covered
    t[0] += 0.3
    assert watch(scene(False), live.pictures) == [tuple(rect)]  # held a moment, no flicker
    t[0] += 1.0
    assert watch(scene(False), live.pictures) == []

    pid = lib.items[0]["id"]
    lib.set(pid, full=True)
    assert PictureWatch()(scene(True), parse({"pictures": lib.active()}).pictures) == [FULL]
    lib.set_game("GTA 6", False)
    assert lib.active() == []
    assert Library(tmp_path / "pictures").items[0]["on"] is False  # kept on disk
    lib.remove(pid)
    assert lib.items == [] and not (tmp_path / "pictures" / f"{pid}.png").exists()


def test_controller_sends_active_pictures(tmp_path, monkeypatch):
    for var in ("LOCALAPPDATA", "APPDATA", "STREAMSAFE_DATA"):
        monkeypatch.setenv(var, str(tmp_path))
    c = controller.Controller()
    c.preview_file().parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(c.preview_file()), np.full((360, 640, 3), 90, np.uint8))
    assert c.pictures.freeze(c.preview_file())
    assert c.pictures.add("Game", "menu", [0.1, 0.1, 0.4, 0.4], True) == ""
    c.write_control()
    pics = parse(json.loads(c.control_file().read_text())).pictures
    assert len(pics) == 1 and pics[0][5] is True
