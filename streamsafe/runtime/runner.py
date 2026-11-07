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
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from streamsafe.analysis.analyzer import Analyzer
from streamsafe.config import Config
from streamsafe.io.obs_link import ObsStateWriter
from streamsafe.io.sinks import FrameSink
from streamsafe.io.sources import FrameSource
from streamsafe.runtime.cuts import CutDetector
from streamsafe.runtime.gate import OutputGate
from streamsafe.runtime.report import device_name, summarize, write_outputs
from streamsafe.runtime.threads import limit_threads
from streamsafe.runtime.worker import AnalyzerWorker
from streamsafe.types import ANALYSIS_SIZE

LATE_MS = 33.0  # a frame shown later than this after its due time counts as late
KEEP_DECISIONS = 60  # frames of decision history kept behind the output position

AnalyzerFactory = Callable[[Config, float], Analyzer]


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
    gate = OutputGate(cfg.render, source.size, keep_log=bool(cfg.output.diag_out), scene=cfg.scene, fps=rt.fps)
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
    report = summarize(gate.stats, analyzer.stats, emitted, n_dec, rt.fps, dict(mode="offline", gpu=device_name()))
    write_outputs(report, cfg.output.stats_out, cfg.output.diag_out, analyzer.diag, gate.log)
    return report


# -------------------------------------------------------------------------------------- realtime
def _run_realtime(cfg: Config, source: FrameSource, sink: FrameSink | None, factory: AnalyzerFactory) -> dict[str, Any]:
    rt = cfg.runtime
    full_res = cfg.eye.full_res_crops  # the worker gets native frames and cuts eye crops from them
    worker = AnalyzerWorker(cfg, factory, _output_scale(source), source.size if full_res else ANALYSIS_SIZE)
    worker.start()
    gate = OutputGate(cfg.render, source.size, keep_log=bool(cfg.output.diag_out), scene=cfg.scene, fps=rt.fps)
    frames: dict[int, tuple[float, Any]] = {}  # frame id -> (capture time, frame)
    lock = threading.Lock()
    capture_done = threading.Event()
    stats: dict[str, list] = collections.defaultdict(list)

    def capture() -> None:
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
                with lock:
                    frames[fid] = (captured, frame)
        except Exception as exc:  # a broken input ends the run; frames already captured are still shown
            stats["source_errors"].append(f"{exc!r:.300}")
        finally:
            capture_done.set()

    def receive() -> None:
        for decision in worker.results():
            gate.add_decision(decision)

    threading.Thread(target=capture, name="capture", daemon=True).start()
    receiver = threading.Thread(target=receive, name="decisions", daemon=True)
    receiver.start()

    obs = ObsStateWriter(cfg.output.obs_state) if cfg.output.obs_state else None
    v = 0
    try:
        while True:
            with lock:
                item = frames.pop(v, None)
            if item is None:
                if capture_done.is_set():
                    with lock:
                        if v not in frames:
                            break
                    continue
                time.sleep(0.001)
                continue
            captured, frame = item
            due = captured + rt.delay
            while time.perf_counter() < due:
                time.sleep(0.0005)
            out = gate.render(v, frame)
            if sink is not None:
                sink.write(out)
            if obs is not None:
                obs.update(v, gate.muted, out is gate.slate or out is gate.card)
            late = 1000 * (time.perf_counter() - due)
            stats["emit_lateness_ms"].append(late)
            stats["late_frames"].append(int(late > LATE_MS))
            gate.forget_before(v - KEEP_DECISIONS)
            v += 1
    finally:
        if obs is not None:
            obs.close()
            stats["obs_write_errors"].append(obs.errors)
        worker.stop()
        receiver.join(timeout=60)
        worker.close()
    gate.finish()

    stats.update(gate.stats)
    gpu = worker.stats.pop("device", ["?"])[0]
    n_dec = len(worker.stats.get("ms_total", []))
    report = summarize(stats, worker.stats, v, n_dec, rt.fps, dict(mode="realtime", gpu=gpu, delay_s=rt.delay))
    write_outputs(report, cfg.output.stats_out, cfg.output.diag_out, worker.diag, gate.log)
    return report
