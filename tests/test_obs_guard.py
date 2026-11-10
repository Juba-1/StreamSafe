"""The OBS guard script (obs/streamsafe_guard.lua) against a fake ``obslua``: mute, slate, fail closed.
Runs only where the ``lupa`` Lua runtime is installed (dev machines); OBS itself is tested by hand."""

from __future__ import annotations

from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")
ROOT = Path(__file__).resolve().parents[1]
GAME = "Capture Card Audio"
SLATE = "StreamSafe Slate"
OUTPUT = "StreamSafe Output"
RAW = "Game Raw"


class FakeObs:
    """Just the obslua calls the script makes."""

    def __init__(self, lua, settings: dict) -> None:
        self.muted = {GAME: False}
        self.slate = False
        self.timer = None
        self.output_width = 1920
        self.raw_visible = False
        self.switched_to = None
        self.panic = None
        self.sources = {GAME, OUTPUT, "Protected"}
        o = lua.eval("{}")
        o.obs_get_source_by_name = lambda n: n if n in self.sources else None
        o.obs_source_set_muted = self._set_muted
        o.obs_source_release = lambda s: None
        o.obs_frontend_get_current_scene = lambda: "scene"
        o.obs_scene_from_source = lambda s: "scene"
        o.obs_scene_find_source_recursive = lambda sc, n: n if n in (SLATE, OUTPUT, RAW) else None
        o.obs_sceneitem_visible = lambda i: {SLATE: self.slate, OUTPUT: True, RAW: self.raw_visible}[i]
        o.obs_sceneitem_set_visible = self._set_visible
        o.timer_add = self._timer_add
        o.timer_remove = lambda f: None
        o.obs_data_get_string = lambda s, k: settings.get(k, "")
        o.obs_data_get_int = lambda s, k: settings[k]
        o.obs_source_get_width = lambda src: self.output_width
        o.obs_frontend_set_current_scene = self._switch
        o.obs_hotkey_register_frontend = self._register
        o.obs_data_get_array = lambda s, k: None
        o.obs_hotkey_load = lambda h, a: None
        o.obs_data_array_release = lambda a: None
        lua.globals().obslua = o

    def _set_muted(self, src, on) -> None:
        self.muted[src] = bool(on)

    def _set_visible(self, item, on) -> None:
        assert item == SLATE
        self.slate = bool(on)

    def _switch(self, scene) -> None:
        self.switched_to = scene

    def _register(self, name, label, callback):
        self.panic = callback
        return "hotkey"

    def _timer_add(self, f, ms) -> None:
        self.timer = f

    @property
    def state(self) -> tuple[bool, bool]:
        return self.muted[GAME], self.slate


def test_guard_script_fails_closed(tmp_path):
    state = tmp_path / "state.txt"
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    settings = {"state_path": str(state), "audio_names": GAME, "slate_name": SLATE, "stale_ms": 500, "output_name": OUTPUT}
    obs = FakeObs(lua, settings)
    lua.execute((ROOT / "obs" / "streamsafe_guard.lua").read_text())
    lua.globals().script_update(None)
    lua.globals().script_load(None)

    def step(line: str | None = None) -> tuple[bool, bool]:
        if line is not None:
            state.write_text(line)
        obs.timer()
        return obs.state

    assert step() == (True, True), "no state file (StreamSafe not running): mute + slate"
    for seq in range(5):
        step(f"1 {seq} 0 0\n")
    assert obs.state == (False, False), "healthy engine: game audio on, no slate"
    assert step("1 5 1 1\n") == (True, False), "the engine asks for the mute (its own scene card is on screen)"
    assert step("1 6 0 0\n") == (False, False)
    obs.muted[GAME] = True  # the streamer mutes the game by hand
    assert step("1 7 0 0\n")[0] is True, "a manual mute is not undone"
    obs.muted[GAME] = False
    for _ in range(4):
        step()
    assert obs.state == (False, False), "frozen for 400 ms: still trusted"
    step()
    step()
    assert obs.state == (True, True), "frozen for 500 ms or more: fail closed"
    assert step("1 8 0 0\n") == (False, False), "recovers when the engine moves again"
    assert step("0 -1 1 1\n") == (True, True), "a clean stop also fails closed"
    assert step("garbage") == (True, True), "an unreadable file fails closed"


def _guard(tmp_path, **extra):
    state = tmp_path / "state.txt"
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    settings = {"state_path": str(state), "audio_names": GAME, "slate_name": SLATE, "stale_ms": 500, "output_name": OUTPUT}
    settings.update(extra)
    obs = FakeObs(lua, settings)
    lua.execute((ROOT / "obs" / "streamsafe_guard.lua").read_text())
    lua.globals().script_update(None)
    lua.globals().script_load(None)
    seq = [0]

    def healthy_tick() -> tuple[bool, bool]:
        seq[0] += 1
        state.write_text(f"1 {seq[0]} 0 0\n")
        obs.timer()
        return obs.state

    return obs, healthy_tick


def test_guard_panic_hotkey_covers_until_pressed_again(tmp_path):
    obs, tick = _guard(tmp_path)
    assert tick() == (False, False)
    obs.panic(True)
    assert tick() == (True, True), "panic: slate + mute even with a healthy engine"
    obs.panic(False)  # key released: still on
    assert tick() == (True, True)
    obs.panic(True)
    assert tick() == (False, False), "pressed again: back to normal"


def test_guard_fails_closed_on_a_wrong_audio_name(tmp_path):
    _, tick = _guard(tmp_path, audio_names="Capture Card Audio 2")
    assert tick()[1] is True, "the named game audio does not exist: its mute would do nothing -> slate"


def test_guard_fails_closed_when_the_output_window_is_empty(tmp_path):
    obs, tick = _guard(tmp_path)
    assert tick() == (False, False)
    obs.output_width = 0  # window closed / minimised: OBS captures nothing
    assert tick() == (True, True)
    obs.output_width = 1920
    assert tick() == (False, False)


def test_guard_takes_the_raw_game_off_air(tmp_path):
    obs, tick = _guard(tmp_path, raw_name=RAW, protected_scene="Protected")
    assert tick() == (False, False)
    obs.raw_visible = True  # the streamer clicked the raw scene / unhid the raw source
    assert tick() == (True, True)
    assert obs.switched_to == "Protected"
