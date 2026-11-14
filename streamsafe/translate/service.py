"""The translation service the frame loop talks to: a worker process, a bounded hand-over, and a lookup that
turns finished Arabic boxes into pixels on the frame being shown.

Frame loop side (never blocks, never raises into the caller):
  * ``observe(fid, frame)``  (capture thread) a tiny picture signature of each text region of EVERY frame (about
    0.1 ms), and now and then a crop of the region is handed to the worker (when the text changes, or the refresh
    time has passed; at most one read per ``translate_min_gap_s``; a full queue drops the read);
  * ``overlay(v, out)``      (output thread, after the gate) the Arabic boxes for frame ``v``: of all finished reads,
    the one nearest in time whose picture looked like frame ``v`` in that region. The frames wait ``delay`` seconds
    in memory, so a read taken when the English appears is there before that frame is shown: the Arabic is on
    the same frames as the English, stays while the same English line stays and goes when it goes. A read that is
    not finished by then simply is not used.
Worker side: ``worker_main`` in its own process (a crash there, even a segfault in a model library, ends only
the worker). Models load in the background: until they are ready nothing is drawn.
"""

from __future__ import annotations

import bisect
import collections
import contextlib
import multiprocessing as mp
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from streamsafe.config import Config
from streamsafe.translate.arabic import Patch, blend
from streamsafe.translate.pipeline import SAME, Crop, Pipeline

REGIONS = {  # (y0, y1, x0, x1) as fractions of the frame; GTA's subtitle band and its top-left help box
    "subtitles": (0.78, 0.98, 0.08, 0.92),
    "help": (0.0, 0.22, 0.0, 0.42),
}
SIG_SHAPE = (32, 256)  # signature: bright pixels of the region at this size
BRIGHT = 170  # subtitle letters are light on dark outline
SIG_SAME = 0.35  # share of differing bright pixels up to which two pictures show "the same text"
SIG_MIN_PIXELS = 12  # fewer bright pixels than this: nothing is written there
KEEP_FRAMES = 600  # frames of signatures kept (older ones can no longer be shown)
KEEP_READS = 64  # finished reads kept
SEARCH_READS = 8  # reads nearest in time that are tried for a frame

Factory = Callable[[Config], Pipeline]
FrameSize = tuple[int, int]


def region_rect(name: str, size: FrameSize) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = REGIONS[name]
    w, h = size
    return round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)


def region_names(cfg: Config) -> list[str]:
    names = [n.strip() for n in cfg.translate.translate_regions.split(",") if n.strip() in REGIONS]
    return names or ["subtitles"]


def signature(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    small = cv2.resize(frame[y0:y1, x0:x1], (SIG_SHAPE[1], SIG_SHAPE[0]), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) > BRIGHT


def sig_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the bright pixels of either picture that are bright in only one: 0 = same text, 1 = different."""
    union = int(np.count_nonzero(a | b))
    if union < SIG_MIN_PIXELS:
        return 0.0  # nothing written in either
    return int(np.count_nonzero(a ^ b)) / union


def has_text_like(sig: np.ndarray) -> bool:
    return int(np.count_nonzero(sig)) >= SIG_MIN_PIXELS


# ----------------------------------------------------------------------------------------------- worker side
def default_factory(cfg: Config) -> Pipeline:
    from pathlib import Path

    from streamsafe.translate.pipeline import load_pipeline

    return load_pipeline(Path(cfg.translate.translate_dir), cfg.translate.translate_threads, cfg.translate.translate_min_conf)


def lower_priority() -> None:
    """The worker yields the CPU to the frame loop: below normal priority (best effort, never raises)."""
    try:
        if os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(10)
    except Exception:
        pass


def worker_main(cfg: Config, factory: Factory, size: FrameSize, jobs: Any, results: Any, stop: Any) -> None:
    lower_priority()
    try:
        from streamsafe.runtime.threads import limit_threads

        limit_threads()
    except Exception:
        pass
    try:
        pipe = factory(cfg)
        pipe.warmup()
    except BaseException as exc:
        results.put(("failed", f"{exc!r:.300}"))
        return
    results.put(("ready", 0))
    while not stop.is_set():
        try:
            job = jobs.get(timeout=0.25)
        except queue.Empty:
            continue
        if job is None:
            break
        with contextlib.suppress(queue.Empty):  # always the newest read: a slow worker skips, it never falls behind
            while True:
                job = jobs.get_nowait()
        if job is None:
            break
        fid, crops = job
        t0 = time.perf_counter()
        try:
            out = pipe(crops, size)
        except Exception as exc:
            results.put(("error", f"{exc!r:.300}"))
            continue
        results.put(
            ("done", fid, out, 1000 * (time.perf_counter() - t0), {k: list(v) for k, v in pipe.ms.items()} if fid % 20 == 0 else None)
        )
        for v in pipe.ms.values():
            del v[:-50]


# --------------------------------------------------------------------------------------- frame loop side
class TranslationService:
    def __init__(
        self, cfg: Config, size: FrameSize, factory: Factory = default_factory, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self.cfg, self.size, self._factory, self._clock = cfg, size, factory, clock
        self.names = region_names(cfg)
        self.rects = {n: region_rect(n, size) for n in self.names}
        self.ready = False
        self.dead = False
        self.stats: dict[str, Any] = collections.defaultdict(
            int,
            dict.fromkeys(
                ("reads_submitted", "reads_done", "reads_dropped", "worker_errors", "frames_overlaid", "observe_errors", "overlay_errors"),
                0,
            ),
        )
        self.errors: list[str] = []
        self.worker_ms: dict[str, list[float]] = {}
        self.read_ms: list[float] = []
        self.observe_ms: collections.deque[float] = collections.deque(maxlen=20000)  # time taken in the frame loop's threads
        self.overlay_ms: collections.deque[float] = collections.deque(maxlen=20000)
        self._lock = threading.Lock()
        self._sigs: collections.OrderedDict[int, dict[str, np.ndarray]] = collections.OrderedDict()
        self._reads: dict[int, dict[str, Any]] = {}  # fid -> {region: [Patch] | SAME}
        self._read_keys: list[int] = []  # sorted fids of finished reads
        self._read_sigs: dict[int, dict[str, np.ndarray]] = {}  # fid -> the signatures of the frame each read was taken from
        self._submitted: dict[int, dict[str, np.ndarray]] = {}  # the same, while the read is with the worker
        self._last_sub: dict[str, tuple[float, np.ndarray | None]] = dict.fromkeys(self.names, (-1000000000.0, None))
        self._last_any = -1e9
        self._ctx = mp.get_context("spawn")
        self._jobs = self._ctx.Queue(maxsize=max(1, cfg.translate.translate_queue))
        self._results = self._ctx.Queue()
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(target=worker_main, args=(cfg, factory, size, self._jobs, self._results, self._stop), daemon=True)
        self._reader = threading.Thread(target=self._collect, name="translate-results", daemon=True)

    def start(self) -> None:
        """Start the worker in the background (never waits for the models)."""
        try:
            self._proc.start()
            self._reader.start()
        except Exception as exc:
            self._fail(f"start: {exc!r:.200}")

    # -- results ------------------------------------------------------------------------------------------
    def _fail(self, why: str) -> None:
        self.dead = True
        self.ready = False
        self.errors.append(why)

    def _collect(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._results.get(timeout=0.5)
            except queue.Empty:
                if not self._proc.is_alive() and not self.dead:
                    self._fail("worker process ended")
                continue
            except Exception as exc:
                self._fail(f"results: {exc!r:.200}")
                return
            kind = msg[0]
            if kind == "ready":
                self.ready = True
            elif kind == "failed":
                self._fail(str(msg[1]))
            elif kind == "error":
                self.stats["worker_errors"] += 1
                if len(self.errors) < 5:
                    self.errors.append(str(msg[1]))
            elif kind == "done":
                self._store(msg[1], msg[2])
                self.stats["reads_done"] += 1
                self.read_ms.append(msg[3])
                if msg[4]:
                    self.worker_ms = msg[4]

    def _store(self, fid: int, per_region: dict[str, Any]) -> None:
        with self._lock:
            merged: dict[str, Any] = {}
            for name, value in per_region.items():
                if isinstance(value, str) and value == SAME:  # unchanged English: the boxes of the last read of this region
                    value = next((self._reads[k][name] for k in reversed(self._read_keys) if name in self._reads[k]), [])
                merged[name] = value
            if fid not in self._reads:
                bisect.insort(self._read_keys, fid)
            self._reads[fid] = merged
            sigs = self._submitted.pop(fid, None)
            if sigs is not None:
                self._read_sigs[fid] = sigs
            while len(self._read_keys) > KEEP_READS:
                old = self._read_keys.pop(0)
                del self._reads[old]
                self._read_sigs.pop(old, None)
            for k in [k for k in self._submitted if k < fid]:  # reads the worker skipped (it takes the newest only)
                del self._submitted[k]

    # -- frame loop: capture side -------------------------------------------------------------------------
    def observe(self, fid: int, frame: np.ndarray) -> None:
        """Called for every captured frame (capture thread). Never raises, never blocks."""
        if self.dead:
            return
        t0 = time.perf_counter()
        try:
            now = self._clock()
            sigs = {n: signature(frame, r) for n, r in self.rects.items()}
            with self._lock:
                self._sigs[fid] = sigs
                while len(self._sigs) > KEEP_FRAMES:
                    self._sigs.popitem(last=False)
            if not self.ready or now - self._last_any < self.cfg.translate.translate_min_gap_s:
                return
            due = False
            for n, sig in sigs.items():
                t_last, sig_last = self._last_sub[n]
                if not has_text_like(sig):  # nothing is written there: nothing to read
                    self._last_sub[n] = (t_last, sig)
                elif sig_last is None or sig_distance(sig, sig_last) > SIG_SAME or now - t_last >= self.cfg.translate.translate_refresh_s:
                    due = True  # new or changed text, or unchanged text that is due for another look
            if not due:
                return
            crops: dict[str, Crop] = {n: (frame[r[1] : r[3], r[0] : r[2]].copy(), r) for n, r in self.rects.items()}
            with self._lock:
                self._submitted[fid] = sigs
            try:
                self._jobs.put_nowait((fid, crops))
            except queue.Full:
                self.stats["reads_dropped"] += 1
                with self._lock:
                    self._submitted.pop(fid, None)
                return
            self._last_any = now
            self.stats["reads_submitted"] += 1
            for n, sig in sigs.items():
                self._last_sub[n] = (now, sig)
        except Exception as exc:
            self.stats["observe_errors"] += 1
            if len(self.errors) < 5:
                self.errors.append(f"observe: {exc!r:.200}")
        finally:
            self.observe_ms.append(1000 * (time.perf_counter() - t0))

    # -- frame loop: output side --------------------------------------------------------------------------
    def patches_for(self, v: int) -> list[Patch]:
        with self._lock:
            sigs = self._sigs.get(v)
            keys = list(self._read_keys)
            if sigs is None or not keys:
                return []
            i = bisect.bisect_left(keys, v)
            near = sorted(keys[max(0, i - SEARCH_READS) : i + SEARCH_READS], key=lambda k: abs(k - v))[:SEARCH_READS]
            reads = [(k, self._reads[k], self._read_sigs.get(k)) for k in near]
        out: list[Patch] = []
        for name, sig in sigs.items():
            for _k, read, read_sigs in reads:
                if name not in read or read_sigs is None:
                    continue
                if sig_distance(sig, read_sigs[name]) <= SIG_SAME:
                    out += read[name]
                    break
        return out

    def overlay(self, v: int, out: np.ndarray) -> np.ndarray:
        """``out`` with the Arabic of frame ``v`` on it: ``out`` itself when there is none, else a COPY (the gate
        keeps ``out`` to repeat it on a held frame; the Arabic must not stick to it). Never raises."""
        if self.dead:
            return out
        t0 = time.perf_counter()
        try:
            patches = self.patches_for(v)
            if not patches:
                return out
            shown = out.copy()
            for p in patches:
                blend(shown, p)
            self.stats["frames_overlaid"] += 1
            return shown
        except Exception as exc:
            self.stats["overlay_errors"] += 1
            if len(self.errors) < 5:
                self.errors.append(f"overlay: {exc!r:.200}")
            return out
        finally:
            self.overlay_ms.append(1000 * (time.perf_counter() - t0))

    def forget_before(self, v: int) -> None:
        with self._lock:
            for k in [k for k in self._sigs if k < v - 5]:
                del self._sigs[k]

    def stop(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            self._jobs.put_nowait(None)
        with contextlib.suppress(Exception):
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
        for q in (self._jobs, self._results):
            with contextlib.suppress(Exception):
                q.cancel_join_thread()
                q.close()

    def report(self) -> dict[str, Any]:
        s = dict(self.stats)
        ms = self.read_ms

        def summary(xs) -> dict[str, float]:
            a = np.asarray(list(xs), np.float64)
            return (
                dict(mean=round(float(a.mean()), 3), p95=round(float(np.percentile(a, 95)), 3), max=round(float(a.max()), 3))
                if len(a)
                else {}
            )

        return dict(
            enabled=True,
            observe_ms=summary(self.observe_ms),
            overlay_ms=summary(self.overlay_ms),
            worker_ms={k: summary(v) for k, v in self.worker_ms.items()},
            ready=self.ready,
            dead=self.dead,
            errors=self.errors[:5],
            read_ms_mean=round(sum(ms) / len(ms), 1) if ms else 0.0,
            **s,
        )
