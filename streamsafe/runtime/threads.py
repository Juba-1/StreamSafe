"""Thread / BLAS limits. A container can show 128 cores while its CPU quota is 16; libraries that
start one thread per visible core then get throttled and stall the renderer (late frames)."""

from __future__ import annotations

import os
import sys


def usable_cpus() -> int:
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows / macOS
        return os.cpu_count() or 4


LIB_THREADS = max(1, min(4, usable_cpus() // 2))
CV_THREADS = 2


def limit_threads() -> None:
    """Call at process start, and again after the models are loaded."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(LIB_THREADS))
    import cv2

    cv2.setNumThreads(CV_THREADS)
    torch = sys.modules.get("torch")  # never import torch just for this (the renderer does not need it)
    if torch is not None:
        torch.set_num_threads(LIB_THREADS)
