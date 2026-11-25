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
import contextlib
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.config import Config
from streamsafe.io import link
from streamsafe.io.obs_link import ObsStateWriter
from streamsafe.io.sinks import FrameSink, takes_audio
from streamsafe.io.sources import FrameSource
from streamsafe.runtime.control import LiveControl
from streamsafe.runtime.cuts import CutDetector
from streamsafe.runtime.gate import OutputGate
from streamsafe.runtime.pictures import PictureWatch
from streamsafe.runtime.report import device_stats, device_summary, summarize, write_outputs
from streamsafe.runtime.threads import limit_threads
from streamsafe.runtime.worker import AnalyzerWorker
from streamsafe.types import ANALYSIS_SIZE

LATE_MS = 33.0  # a frame shown later than this after its due time counts as late
KEEP_DECISIONS = 60  # frames of decision history kept behind the output position

AnalyzerFactory = Callable[[Config, float], Analyzer]


def start_translation(cfg: Config, size: tuple[int, int], factory: Any = None) -> Any:
    """The English -> Arabic overlay service (streamsafe/translate), or None: off (the default), models not
    installed, or any failure while starting. Nothing here can stop the stream: whatever goes wrong, the run is
    exactly the run without translation."""
    if not cfg.translate.translate_ar:
        return None
    try:
        from streamsafe.translate import pack
        from streamsafe.translate.service import TranslationService, default_factory

        if factory is None:
            why = pack.problem(cfg.translate.translate_dir, full=True)
            if why:
                print(f"TRANSLATE_OFF {why}", flush=True)
                return None
        service = TranslationService(cfg, size, factory or default_factory)
        service.start()
        return service
    except Exception as exc:
        print(f"TRANSLATE_OFF {exc!r:.200}", flush=True)
        return None


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


COVER_BGR = (34, 14, 24)  # the always-covered parts: the window's own night colour, opaque


def cover_parts(frame: np.ndarray, parts: tuple) -> None:
    """Paint the streamer's always-covered parts (fractions 0-1) opaque, in place."""
    h, w = frame.shape[:2]
    for x0, y0, x1, y1 in parts:
        frame[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)] = COVER_BGR


class Snapper:
    """``snap`` on its own thread: the resize and the PNG never delay a frame of the stream. Busy = skipped."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._busy = threading.Event()

    def __call__(self, frame: np.ndarray, boxes: list | None = None) -> None:
        if self._busy.is_set():
            return
        self._busy.set()

        def run() -> None:
            try:
                snap(frame, self.path)
                tmp = self.path + ".json.tmp"  # where the person covers are, so the window can offer to release one
                with contextlib.suppress(OSError):
                    Path(tmp).write_text(json.dumps({"t": time.time(), "boxes": [list(map(float, b)) for b in boxes or []]}))
                    os.replace(tmp, self.path + ".json")
            finally:
                self._busy.clear()

        threading.Thread(target=run, name="preview", daemon=True).start()


def snap(frame: np.ndarray, path: str) -> None:
    """A small copy of the shown picture, replaced whole (the window never reads half a file)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (640, max(2, round(640 * h / w))), interpolation=cv2.INTER_AREA)
    tmp = path + ".tmp.png"
    if cv2.imwrite(tmp, small):
        with contextlib.suppress(OSError):
            os.replace(tmp, path)


def default_analyzer(cfg: Config, output_scale: float) -> Analyzer:
    from streamsafe.models import build_models

    return Analyzer(cfg, build_models(cfg), output_scale)


def run(
    cfg: Config,
    source: FrameSource,
    sink: FrameSink | None,
    factory: AnalyzerFactory = default_analyzer,
    translate_factory: Any = None,
) -> dict[str, Any]:
    limit_threads()
    mode = cfg.runtime.mode
    if mode == "realtime":
        return _run_realtime(cfg, source, sink, factory, translate_factory)
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


def _run_realtime(
    cfg: Config, source: FrameSource, sink: FrameSink | None, factory: AnalyzerFactory, translate_factory: Any = None
) -> dict[str, Any]:
    rt = cfg.runtime
    full_res = cfg.eye.full_res_crops  # the worker gets native frames and cuts eye crops from them
    worker = _Workers(lambda: AnalyzerWorker(cfg, factory, _output_scale(source), source.size if full_res else ANALYSIS_SIZE))
    gate = OutputGate(cfg.render, source.size, keep_log=bool(cfg.output.diag_out), scene=cfg.scene, fps=rt.fps, brand=cfg.brand)
    tr = start_translation(cfg, source.size, translate_factory)  # None unless the Arabic text translation is on and ready to load
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
                if tr is not None:
                    tr.observe(fid, frame)  # reads the raw frame now; never blocks, never raises
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
        while not worker.ready.wait(0.5):
            if worker.stopping.is_set() or worker.failed is not None:
                return
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
    ctl = LiveControl.from_env()  # panic key and webcam zones, changed by the window while live
    preview_file = os.environ.get("PREVIEW_FILE", "")
    preview = Snapper(preview_file) if preview_file else None
    parts: tuple = ()  # the always-covered parts, from the live control file
    pics: tuple = ()  # saved pictures covered whenever they show
    watch = PictureWatch()
    preview_every = max(1, round(2 * cfg.runtime.fps))
    try:
        try:
            while True:
                if worker.failed is not None:  # the models never loaded: end the run and say why
                    raise worker.failed
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
                if v % 6 == 0:
                    now_live = ctl.get()
                    gate.manual, gate.zones, parts, pics = now_live.panic, now_live, now_live.covers, now_live.pictures
                found = watch(frame, pics) if pics else []  # on the untouched frame, before any cover is drawn
                out = gate.render(v, frame)
                if (parts or found) and out is not gate.slate and out is not gate.card:
                    cover_parts(out, (*parts, *found))  # always covered (a map, a code), whatever the analysis said
                # Arabic text, AFTER the gate and on top of the cover: on a normally composed frame and on the full-scene
                # cover (card / slate: the dialogue goes on while the picture is covered), never on a held repeat; always
                # on a copy, so nothing the gate keeps (its slate, card or last frame) ever carries the text
                text_ok = out is frame or out is gate.card or out is gate.slate
                visible = tr.overlay(v, out) if tr is not None and text_ok else out
                if preview is not None and v % preview_every == 0:  # what the viewers see, for the window
                    preview(visible, gate.shown_boxes)
                if sink is not None and takes_audio(sink):
                    # fail closed: sound only for a frame that is shown, not muted and not full-screen, and only
                    # that frame's own samples; anything else (also "not arrived") is silence
                    sound = None
                    if sound_of is not None and not gate.muted and out is not gate.slate and out is not gate.card:
                        sound = sound_of(number)
                        n_sound += sound is not None
                        n_no_sound += sound is None
                    sink.write(visible, sound)  # type: ignore[call-arg]
                elif sink is not None:
                    sink.write(visible)
                if obs is not None:
                    obs.update(v, gate.muted, out is gate.slate or out is gate.card)
                late = 1000 * (time.perf_counter() - due)
                stats["emit_lateness_ms"].append(late)
                stats["late_frames"].append(int(late > LATE_MS))
                gate.forget_before(v - KEEP_DECISIONS)
                if tr is not None:
                    if late > LATE_MS:
                        tr.stream_late()
                    tr.forget_before(v)
                live.tick(v, late, waiting, len(stats["dropped_frames"]))
                v += 1
                shown += 1
        except KeyboardInterrupt:  # Ctrl+C: stop showing frames, but still close cleanly and write the report
            stopped_by_user = True
            print("STOP_PHASE frames stopped", flush=True)
    finally:
        if obs is not None:
            obs.close()
            stats["obs_write_errors"].append(obs.errors)
        worker.shutdown(receiver)
        print("STOP_PHASE analyzer stopped", flush=True)
        if tr is not None:
            tr.stop()
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
        late_frames=sum(stats.get("late_frames", [])),
        picture_frames=watch.matches,
        audio_frames=n_sound,
        audio_missing_frames=n_no_sound if getattr(source, "has_audio", False) else 0,
    )
    if tr is not None:
        extra["translate"] = tr.report()
    report = summarize(stats, last.stats, shown, n_dec, rt.fps, extra)
    write_outputs(report, cfg.output.stats_out, cfg.output.diag_out, last.diag, gate.log)
    return report


class _Workers:
    """The current analyzer process, replaced when it dies or keeps failing. ``submit`` always goes to the
    current one; while a replacement loads, frames get no decisions and the gate shows the slate."""

    def __init__(self, make: Callable[[], AnalyzerWorker]) -> None:
        self._make = make
        self.stopping = threading.Event()
        self.ready = threading.Event()  # the first analyzer has loaded its models
        self.failed: BaseException | None = None  # the first analyzer could not load: the run ends with this
        self._last_restart = -1e9
        self._lock = threading.Lock()
        self._accepting = False
        self.current = make()
        # the models load while the stream already runs: viewers get the slate (no decisions yet) and the platform
        # is connected at once, instead of a window stuck on "starting" while a first load takes minutes
        threading.Thread(target=self._first_start, name="analyzer-load", daemon=True).start()

    def _first_start(self) -> None:
        t0 = time.monotonic()
        link.status("protection_loading")
        try:
            self.current.start()
        except BaseException as exc:
            if not self.stopping.is_set():
                self.failed = exc
            return
        with self._lock:
            self._accepting = not self.stopping.is_set()
        link.status("protection_ready", seconds=round(time.monotonic() - t0, 1))
        self.ready.set()

    def submit(self, frame_id: int, frame: np.ndarray, cut: bool) -> None:
        with self._lock:  # never write into a ring that is being released
            if self._accepting:
                self.current.submit(frame_id, frame, cut)

    def shutdown(self, receiver: threading.Thread) -> None:
        self.stopping.set()
        stopped = self.current
        # the final stats come right after the analyzer's loop ends; its process exit (freeing the GPU models) can
        # take long on a busy card, and nothing waits for that: after a short grace it is ended
        stopped.signal_stop()
        receiver.join(timeout=10)
        stopped.stop(timeout=3.0)
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
