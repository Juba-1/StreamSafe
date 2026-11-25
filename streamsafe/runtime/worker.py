"""The analyzer in its own process.

Why a process: the renderer must hit its 33 ms deadline every frame; sharing one interpreter
(GIL) with the analyzer caused bursts of late frames on busy CPUs. Frames go to the worker
through a shared-memory ring (no pickling of images); the worker always takes the NEWEST frame,
skipping older ones, so a slow analyzer lowers the decision rate instead of building a backlog. A fast one
is held to one decision every ``step_every`` frames.
Decisions come back through a queue.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import queue
import sys
import time
from collections.abc import Callable, Iterator
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from streamsafe.config import Config
from streamsafe.runtime.report import device_stats
from streamsafe.runtime.threads import limit_threads
from streamsafe.types import ANALYSIS_SIZE, Decision

N_SLOTS = 16
N_CUT_FLAGS = 1 << 16  # cut flags kept per frame id (ring)
READY_TIMEOUT_S = 900.0  # first model download can be slow
_END = "END"

AnalyzerFactory = Callable[[Config, float], Callable]  # (cfg, output_scale) -> analyzer(frame_id, frame, cut)


class AnalyzerWorker:
    def __init__(self, cfg: Config, factory: AnalyzerFactory, output_scale: float, frame_size: tuple[int, int] = ANALYSIS_SIZE) -> None:
        w, h = frame_size
        ctx = mp.get_context("spawn")  # CUDA cannot be forked
        self._shm = shared_memory.SharedMemory(create=True, size=N_SLOTS * h * w * 3)
        self._ring = np.ndarray((N_SLOTS, h, w, 3), np.uint8, buffer=self._shm.buf)
        self._slot_fid = ctx.Array("q", [-1] * N_SLOTS, lock=False)
        self._latest = ctx.Value("q", -1, lock=False)
        self._cuts = ctx.Array("b", N_CUT_FLAGS, lock=False)
        self._done, self._ready, self._out = ctx.Event(), ctx.Event(), ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(
                *(cfg, factory, output_scale, frame_size, self._shm.name),
                *(self._slot_fid, self._latest, self._cuts, self._done, self._ready, self._out),
            ),
            daemon=True,
        )
        self.stats: dict = {}
        self.diag: list = []

    def start(self) -> None:
        self._proc.start()
        deadline = time.monotonic() + READY_TIMEOUT_S
        while not self._ready.wait(1.0):
            if not self._proc.is_alive():
                raise RuntimeError("analyzer process died while loading its models")
            if time.monotonic() > deadline:
                raise TimeoutError("analyzer process did not become ready")

    def submit(self, frame_id: int, frame: np.ndarray, cut: bool) -> None:
        """Hand a BGR frame of ``frame_size`` to the worker (never blocks)."""
        slot = frame_id % N_SLOTS
        self._slot_fid[slot] = -1  # invalidate while writing
        self._ring[slot] = frame
        self._slot_fid[slot] = frame_id
        self._cuts[frame_id % N_CUT_FLAGS] = int(cut)
        self._latest.value = frame_id

    def results(self) -> Iterator[Decision]:
        """Decisions as they arrive; ends after ``stop`` once the worker has reported its stats."""
        while True:
            try:
                item = self._out.get(timeout=1.0)
            except queue.Empty:
                if not self._proc.is_alive():
                    return  # crashed without reporting: the gate fails closed on missing decisions
                continue
            if isinstance(item, tuple) and item[0] == _END:
                self.stats, self.diag = item[1], item[2]
                return
            yield item

    def signal_stop(self) -> None:
        """Tell the analyzer to finish: it sends its final stats, then ends (``stop`` waits for that)."""
        self._done.set()

    def stop(self, timeout: float = 30.0) -> None:
        self._done.set()
        self._proc.join(timeout=timeout)
        if self._proc.is_alive():
            self._proc.terminate()

    def close(self) -> None:
        self._shm.close()
        with contextlib.suppress(FileNotFoundError):
            self._shm.unlink()


def _worker_main(
    cfg: Config,
    factory: AnalyzerFactory,
    output_scale: float,
    frame_size: tuple[int, int],
    shm_name: str,
    slot_fid: Any,  # multiprocessing shared array / value / event / queue proxies
    latest: Any,
    cuts: Any,
    done: Any,
    ready: Any,
    out: Any,
) -> None:
    limit_threads()
    t0 = time.monotonic()
    analyzer = factory(cfg, output_scale)
    limit_threads()  # again: the models may have imported torch
    t1 = time.monotonic()
    print(f"STREAMSAFE_LOAD models {t1 - t0:.1f} s", file=sys.stderr, flush=True)  # in the report: where a slow start went
    analyzer.warmup()
    print(f"STREAMSAFE_LOAD warmup {time.monotonic() - t1:.1f} s", file=sys.stderr, flush=True)
    slow = getattr(getattr(analyzer, "headroom", None), "warning", lambda: "")()
    if slow:  # below the decision rate: said at start, not only in the final report
        print(f"STREAMSAFE_SLOW {slow}", file=sys.stderr, flush=True)
    ready.set()
    w, h = frame_size
    shm = shared_memory.SharedMemory(name=shm_name)
    ring = np.ndarray((N_SLOTS, h, w, 3), np.uint8, buffer=shm.buf)
    last = -1
    step = max(1, cfg.runtime.step_every)
    parent = mp.parent_process()
    try:
        while not done.is_set():
            fid = latest.value
            # at most one decision every ``step`` frames (10/s at 30 fps), as in the offline benchmarks: the
            # policy counts decisions (release, hand-off, coasting), so a faster GPU must not change what it does,
            # and the streamer's GPU is not kept busy for nothing (OBS shares it)
            if fid < last + step:
                if parent is not None and not parent.is_alive():
                    break  # the engine was killed (Task Manager): free the GPU instead of lingering
                time.sleep(0.001)
                continue
            slot = fid % N_SLOTS
            if slot_fid[slot] != fid:
                continue
            frame = ring[slot].copy()
            if slot_fid[slot] != fid:
                continue  # overwritten while copying: take the next one
            cut = last < 0 or any(cuts[i % N_CUT_FLAGS] for i in range(last + 1, fid + 1))
            last = fid
            out.put(analyzer(fid, frame, cut))
    finally:
        stats = dict(getattr(analyzer, "stats", {}))
        stats["vram_gb"] = [_peak_vram_gb()]
        device_stats(stats)
        out.put((_END, stats, getattr(analyzer, "diag", [])))
        del ring
        shm.close()


def _peak_vram_gb() -> float:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)
