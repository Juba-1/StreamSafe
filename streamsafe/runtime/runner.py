"""Runs the engine from a source to a sink.

realtime (the product): the analyzer runs in its own process and works on the newest frame it
    can get; every frame is shown exactly ``delay`` seconds after it was captured, protected with
    whatever decisions exist by then (fail closed otherwise).
offline (quality benchmark): every ``step_every``-th frame is analysed, no clock - the result is
    the same on any machine, so quality changes can be measured without host noise.
Both modes share the analyzer, the output gate and the reporting.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.config import Config
from streamsafe.io.obs_link import ObsStateWriter
from streamsafe.io.sinks import FrameSink, takes_audio
from streamsafe.io.sources import FrameSource
from streamsafe.runtime.cuts import CutDetector
from streamsafe.runtime.gate import OutputGate
from streamsafe.runtime.report import device_stats, device_summary, summarize, write_outputs
from streamsafe.runtime.threads import limit_threads
from streamsafe.runtime.worker import AnalyzerWorker
from streamsafe.types import ANALYSIS_SIZE

LATE_MS = 33.0  # a frame shown later than this after its due time counts as late
KEEP_DECISIONS = 60  # frames of decision history kept behind the output position

AnalyzerFactory = Callable[[Config, float], Analyzer]


class LiveLog:
    """Prints ``LIVE {...}`` every ``every_s`` seconds (0 = off): output position, worst lateness and
    the frames waiting in memory since the last line, frames dropped so far."""

    def __init__(self, every_s: float) -> None:
        self.every_s, self._next, self._worst, self._waiting = every_s, time.perf_counter() + every_s, 0.0, 0

    def tick(self, frame_id: int, late_ms: float, waiting: int, dropped: int) -> None:
        if self.every_s <= 0:
            return
        self._worst, self._waiting = max(self._worst, late_ms), max(self._waiting, waiting)
        if time.perf_counter() >= self._next:
            line = dict(frame=frame_id, late_ms_max=round(self._worst, 1), waiting_max=self._waiting, dropped=dropped)
            print("LIVE " + json.dumps(line), flush=True)
            self._next, self._worst, self._waiting = self._next + self.every_s, 0.0, 0


def default_analyzer(cfg: Config, output_scale: float) -> Analyzer:
    from streamsafe.models import build_models

    return Analyzer(cfg, build_models(cfg), output_scale)


def run(cfg: Config, source: FrameSource, sink: FrameSink | None, factory: AnalyzerFactory = default_analyzer) -> dict[str, Any]:
    limit_threads()
    mode = cfg.runtime.mode
    if mode == "realtime":
        return _run_realtime(cfg, source, sink, factory)
    if mode == "offline":
        return _run_offline(cfg, source, sink, factory)
    raise ValueError(f"unknown mode: {mode!r} (realtime | offline)")


def _output_scale(source: FrameSource) -> float:
    return source.size[0] / ANALYSIS_SIZE[0]


def _to_analysis(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------------------- offline
def _run_offline(cfg: Config, source: FrameSource, sink: FrameSink | None, factory: AnalyzerFactory) -> dict[str, Any]:
    rt = cfg.runtime
    analyzer = factory(cfg, _output_scale(source))
    analyzer.warmup()
    gate = OutputGate(cfg.render, source.size, keep_log=bool(cfg.output.diag_out), scene=cfg.scene, fps=rt.fps, brand=cfg.brand)
    cuts = CutDetector()
    pending: collections.OrderedDict = collections.OrderedDict()  # frames waiting for their decisions
    cut_since_last, emitted = True, 0
    # a frame is final once every decision that may protect it exists
    horizon = cfg.render.max_gap + rt.step_every
    if cfg.render.lookahead:  # frames also wait like the realtime delay, so look-ahead can reach them
        horizon = max(horizon, cfg.render.lookahead_frames + rt.step_every)

    def emit(v: int, frame: np.ndarray) -> None:
        nonlocal emitted
        out = gate.render(v, frame)
        if sink is not None:
            sink.write(out)
        emitted += 1

    for fid, frame in enumerate(source.frames()):
        cut = cuts(frame)
        gate.mark_cut(fid, cut)
        cut_since_last |= cut
        pending[fid] = frame
        if fid % rt.step_every == 0:
            gate.add_decision(analyzer(fid, frame, cut_since_last, now=fid / rt.fps))
            cut_since_last = False
        while pending and next(iter(pending)) <= fid - horizon:
            emit(*pending.popitem(last=False))
            gate.forget_before(emitted - KEEP_DECISIONS)
    while pending:
        emit(*pending.popitem(last=False))
    gate.finish()

    n_dec = len(analyzer.stats.get("ms_total", []))
    device_stats(analyzer.stats)
    extra = dict(mode="offline", **device_summary(analyzer.stats))
    report = summarize(gate.stats, analyzer.stats, emitted, n_dec, rt.fps, extra)
    write_outputs(report, cfg.output.stats_out, cfg.output.diag_out, analyzer.diag, gate.log)
    return report


# -------------------------------------------------------------------------------------- realtime
RESTART_AFTER_FAILS = 30  # consecutive failed decisions (3 s at 10/s) -> a new analyzer process
RESTART_BACKOFF_S = 10.0


def _run_realtime(cfg: Config, source: FrameSource, sink: FrameSink | None, factory: AnalyzerFactory) -> dict[str, Any]:
    rt = cfg.runtime
    full_res = cfg.eye.full_res_crops  # the worker gets native frames and cuts eye crops from them
    worker = _Workers(lambda: AnalyzerWorker(cfg, factory, _output_scale(source), source.size if full_res else ANALYSIS_SIZE))
    gate = OutputGate(cfg.render, source.size, keep_log=bool(cfg.output.diag_out), scene=cfg.scene, fps=rt.fps, brand=cfg.brand)
    frames: dict[int, tuple[float, Any, int]] = {}  # frame id -> (capture time, frame, its number in the source)
    # sound travels with its frame: taken from the source by the frame's own number when the frame is shown
    sound_of = getattr(source, "audio_chunk", None) if takes_audio(sink) else None
    if sound_of is not None and hasattr(source, "audio_keep"):
        source.audio_keep(rt.delay + rt.max_late_s + 2.0)  # sound is kept as long as a frame can wait
    n_sound = n_no_sound = 0
    cap = int((rt.delay + rt.max_late_s + 1.0) * rt.fps)  # frames held at most, even if the output stalls
    n_captured = 0  # frame ids below this have been captured (a missing one was dropped)
    lock = threading.Lock()
    capture_done = threading.Event()
    stats: dict[str, list] = collections.defaultdict(list)

    def capture() -> None:
        nonlocal n_captured
        cuts = CutDetector()
        t0 = time.perf_counter() + 0.5
        try:
            for fid, frame in enumerate(source.frames()):
                if source.live:
                    captured = time.perf_counter()
                else:  # pace a file like a live feed
                    captured = t0 + fid / rt.fps
                    while time.perf_counter() < captured:
                        time.sleep(0.0005)
                cut = cuts(frame)
                gate.mark_cut(fid, cut)
                worker.submit(fid, frame if full_res else _to_analysis(frame), cut)
                number = source.frame_index if sound_of is not None else fid  # type: ignore[attr-defined]
                with lock:
                    frames[fid] = (captured, frame, number)
                    n_captured = fid + 1
                    if len(frames) > cap:  # output stalled: forget the oldest, keep memory bounded
                        del frames[min(frames)]
        except Exception as exc:  # a broken input ends the run; frames already captured are still shown
            stats["source_errors"].append(f"{exc!r:.300}")
        finally:
            capture_done.set()

    def receive() -> None:
        while True:
            failed = 0
            for decision in worker.current.results():
                gate.add_decision(decision)
                failed = 0 if decision.ok else failed + 1
                if failed >= RESTART_AFTER_FAILS:
                    break  # e.g. a lost GPU device: every decision fails
            if worker.stopping.is_set():
                return
            # the analyzer died or keeps failing: viewers get the slate (no decisions) while a new one loads
            stats["worker_restarts"].append(round(time.perf_counter(), 1))
            worker.restart(stats)

    threading.Thread(target=capture, name="capture", daemon=True).start()
    receiver = threading.Thread(target=receive, name="decisions", daemon=True)
    receiver.start()

    obs = ObsStateWriter(cfg.output.obs_state) if cfg.output.obs_state else None
    live = LiveLog(rt.live_log_s)
    v = shown = 0
    stopped_by_user = False
    try:
        try:
            while True:
                with lock:
                    item = frames.pop(v, None)
                    waiting, gone = len(frames), v < n_captured
                if item is None and gone:  # dropped while waiting (output stalled)
                    stats["dropped_frames"].append(v)
                    v += 1
                    continue
                if item is None:
                    if capture_done.is_set():
                        with lock:
                            if v not in frames:
                                break
                        continue
                    time.sleep(0.001)
                    continue
                captured, frame, number = item
                due = captured + rt.delay
                if time.perf_counter() - due > rt.max_late_s:
                    stats["dropped_frames"].append(v)  # never shown: skipping it costs a stutter, never safety
                    gate.forget_before(v - KEEP_DECISIONS)
                    v += 1
                    continue
                while time.perf_counter() < due:
                    time.sleep(0.0005)
                out = gate.render(v, frame)
                if sink is not None and takes_audio(sink):
                    # fail closed: sound only for a frame that is shown, not muted and not full-screen, and only
                    # that frame's own samples; anything else (also "not arrived") is silence
                    sound = None
                    if sound_of is not None and not gate.muted and out is not gate.slate and out is not gate.card:
                        sound = sound_of(number)
                        n_sound += sound is not None
                        n_no_sound += sound is None
                    sink.write(out, sound)  # type: ignore[call-arg]
                elif sink is not None:
                    sink.write(out)
                if obs is not None:
                    obs.update(v, gate.muted, out is gate.slate or out is gate.card)
                late = 1000 * (time.perf_counter() - due)
                stats["emit_lateness_ms"].append(late)
                stats["late_frames"].append(int(late > LATE_MS))
                gate.forget_before(v - KEEP_DECISIONS)
                live.tick(v, late, waiting, len(stats["dropped_frames"]))
                v += 1
                shown += 1
        except KeyboardInterrupt:  # Ctrl+C: stop showing frames, but still close cleanly and write the report
            stopped_by_user = True
    finally:
        if obs is not None:
            obs.close()
            stats["obs_write_errors"].append(obs.errors)
        worker.shutdown(receiver)
    gate.finish()

    stats.update(gate.stats)
    last = worker.current
    device = device_summary(last.stats)
    n_dec = len(last.stats.get("ms_total", []))
    restarts = len(stats.get("worker_restarts", []))
    extra = dict(
        mode="realtime",
        **device,
        delay_s=rt.delay,
        frames_in=v,
        dropped_frames=v - shown,
        worker_restarts=restarts,
        stopped_by_user=stopped_by_user,
        # frames sent with their sound / frames that should have had sound but none was there (silence instead)
        audio_frames=n_sound,
        audio_missing_frames=n_no_sound if getattr(source, "has_audio", False) else 0,
    )
    report = summarize(stats, last.stats, shown, n_dec, rt.fps, extra)
    write_outputs(report, cfg.output.stats_out, cfg.output.diag_out, last.diag, gate.log)
    return report


class _Workers:
    """The current analyzer process, replaced when it dies or keeps failing. ``submit`` always goes to the
    current one; while a replacement loads, frames get no decisions and the gate shows the slate."""

    def __init__(self, make: Callable[[], AnalyzerWorker]) -> None:
        self._make = make
        self.stopping = threading.Event()
        self._last_restart = -1e9
        self._lock = threading.Lock()
        self._accepting = True
        self.current = make()
        self.current.start()

    def submit(self, frame_id: int, frame: np.ndarray, cut: bool) -> None:
        with self._lock:  # never write into a ring that is being released
            if self._accepting:
                self.current.submit(frame_id, frame, cut)

    def shutdown(self, receiver: threading.Thread) -> None:
        self.stopping.set()
        stopped = self.current
        stopped.stop()
        receiver.join(timeout=60)  # it collects the analyzer's final stats
        with self._lock:
            self._accepting = False
        if self.current is not stopped:  # a replacement finished loading while we were stopping
            self.current.stop()
        self.current.close()

    def restart(self, stats: dict[str, list]) -> None:
        with self._lock:
            self._accepting = False
        old = self.current
        old.stop(timeout=5.0)
        old.close()
        if time.monotonic() - self._last_restart < 60.0:  # failing again soon: do not spin, slate meanwhile
            self.stopping.wait(RESTART_BACKOFF_S)
        self._last_restart = time.monotonic()
        while not self.stopping.is_set():
            new = self._make()
            try:
                new.start()
            except Exception as exc:  # models fail to load again: keep the slate, try again later
                stats["worker_start_errors"].append(f"{exc!r:.200}")
                new.stop(timeout=5.0)
                new.close()
                self.stopping.wait(RESTART_BACKOFF_S)
                continue
            with self._lock:
                self.current = new
                self._accepting = True
            return
