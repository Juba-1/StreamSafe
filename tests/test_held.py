"""The held engine (app/held.py): models loaded ahead, the run waits for Start."""

import io
import json
import typing

from streamsafe import cli
from streamsafe.app import engine, held
from streamsafe.io import link
from streamsafe.runtime.cover_all import cover_all_analyzer
from tests.test_stream_flow import _clip


def test_wait_for_go_reads_the_window_s_start(monkeypatch):
    monkeypatch.delenv(engine.STOP_ENV, raising=False)
    assert engine.wait_for_go(io.StringIO("go\n")) is True
    assert engine.wait_for_go(io.StringIO("")) is False  # the window closed: end, nothing opened
    assert engine.wait_for_go(io.StringIO("stop\n")) is False
    assert engine.wait_for_go(io.StringIO("noise\nGO\n")) is True


def _args(clip, out, folder) -> list[str]:
    return ["--input", f"file:{clip}", "--output", f"record:{out}", "--mode", "realtime", "--size", "320x180", "--report-dir", str(folder)]


def test_a_held_run_loads_first_then_runs_on_go(tmp_path, capsys):
    """hold() is asked only after the analyzer started loading; on True the run goes on as any other."""
    clip = _clip(tmp_path / "game.mp4")
    out = tmp_path / "protected.mkv"
    asked = []
    said = []
    link._sink = said.append
    try:
        code = cli.main(
            _args(clip, out, tmp_path),
            factory=cover_all_analyzer,
            hold=lambda: asked.append([json.loads(x[len(link.PREFIX) :])["event"] for x in said]) or True,
        )
    finally:
        link._sink = None
    assert code == 0 and out.is_file()
    assert asked and "protection_loading" in asked[0], "the models load before Start is waited for"
    names = [json.loads(x[len(link.PREFIX) :])["event"] for x in said]
    assert names.index("protection_ready") < names.index("stopped")


def test_a_held_engine_that_is_not_started_opens_nothing(tmp_path):
    clip = _clip(tmp_path / "game.mp4")
    out = tmp_path / "protected.mkv"
    code = cli.main(
        _args(clip, out, tmp_path),
        factory=cover_all_analyzer,
        hold=lambda: False,
    )
    assert code == 0 and not out.exists() and not list(tmp_path.glob("session_*.txt")), "no output, no report"


class _Proc:
    """Stands in for engine.EngineProcess."""

    made: typing.ClassVar[list] = []

    def __init__(self, on_line, redactor=None) -> None:
        self.on_line, self.alive, self.stop_requested, self.sent = on_line, False, False, []
        _Proc.made.append(self)

    def start(self, cmd, env, log_path=None) -> None:
        self.cmd, self.env, self.alive = cmd, env, True

    @property
    def running(self) -> bool:
        return self.alive

    def stop(self, wait_s: float = 0) -> None:
        self.stop_requested, self.alive = True, False


def test_held_starts_matches_and_replaces(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "EngineProcess", _Proc)
    _Proc.made = []
    now = [100.0]
    h = held.Held(clock=lambda: now[0])
    log = lambda: tmp_path / "x.log"  # noqa: E731
    cmd, env = ["run", "--input", "srt"], {"A": "1", "THR_SCALE": "1.95"}
    h.want(cmd, env, (), log)
    assert not _Proc.made, "waits for the settings to settle"
    now[0] += held.SETTLE_S
    h.want(cmd, env, (), log)
    assert len(_Proc.made) == 1 and _Proc.made[0].env[engine.HOLD_ENV] == "1"
    h.want(cmd, {**env, "THR_SCALE": "2.4"}, (), log)
    assert len(_Proc.made) == 1, "the strength is read live: no new engine"
    first = _Proc.made[0]
    first.on_line('STATUS {"event": "protection_ready"}')
    assert h.ready
    taken = h.take(cmd, env)
    assert taken is not None and taken[0] is first and any("protection_ready" in x for x in taken[1])
    assert h.take(cmd, env) is None, "taken once"
    h.want(["run", "--input", "other"], env, (), log)  # a changed setting
    now[0] += held.SETTLE_S
    h.want(["run", "--input", "other"], env, (), log)
    second = _Proc.made[-1]
    assert second is not first and second.cmd == ["run", "--input", "other"]
    assert h.take(cmd, env) is None and second.stop_requested, "Start with other settings ends the held one"


def test_held_that_ends_by_itself_is_retried_later(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "EngineProcess", _Proc)
    _Proc.made = []
    now = [100.0]
    h = held.Held(clock=lambda: now[0])
    log = lambda: tmp_path / "x.log"  # noqa: E731
    h.want(["a"], {}, (), log)
    now[0] += held.SETTLE_S
    h.want(["a"], {}, (), log)
    assert len(_Proc.made) == 1
    _Proc.made[0].alive = False  # could not load
    h.want(["a"], {}, (), log)
    assert len(_Proc.made) == 1, "not again at once"
    now[0] += held.RETRY_S
    h.want(["a"], {}, (), log)
    assert len(_Proc.made) == 2
    h.drop(wait=True)
    assert _Proc.made[1].stop_requested and not h.ready and not h.loading
