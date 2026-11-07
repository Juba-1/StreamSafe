"""The OBS guard script (obs/streamsafe_guard.lua) against a fake ``obslua``: mute, slate, fail closed.
Runs only where the ``lupa`` Lua runtime is installed (dev machines); OBS itself is tested by hand."""

from __future__ import annotations

from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")
ROOT = Path(__file__).resolve().parents[1]
GAME = "Capture Card Audio"
SLATE = "StreamSafe Slate"


class FakeObs:
    """Just the obslua calls the script makes."""

    def __init__(self, lua, settings: dict) -> None:
        self.muted = {GAME: False}
        self.slate = False
        self.timer = None
        o = lua.eval("{}")
        o.obs_get_source_by_name = lambda n: n if n in self.muted else None
        o.obs_source_set_muted = self._set_muted
        o.obs_source_release = lambda s: None
        o.obs_frontend_get_current_scene = lambda: "scene"
        o.obs_scene_from_source = lambda s: "scene"
        o.obs_scene_find_source_recursive = lambda sc, n: "item" if n == SLATE else None
        o.obs_sceneitem_visible = lambda i: self.slate
        o.obs_sceneitem_set_visible = self._set_visible
        o.timer_add = self._timer_add
        o.timer_remove = lambda f: None
        o.obs_data_get_string = lambda s, k: settings[k]
        o.obs_data_get_int = lambda s, k: settings[k]
        lua.globals().obslua = o

    def _set_muted(self, src, on) -> None:
        self.muted[src] = bool(on)

    def _set_visible(self, item, on) -> None:
        self.slate = bool(on)

    def _timer_add(self, f, ms) -> None:
        self.timer = f

    @property
    def state(self) -> tuple[bool, bool]:
        return self.muted[GAME], self.slate


def test_guard_script_fails_closed(tmp_path):
    state = tmp_path / "state.txt"
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    settings = {"state_path": str(state), "audio_names": f"{GAME}, Other", "slate_name": SLATE, "stale_ms": 500}
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
