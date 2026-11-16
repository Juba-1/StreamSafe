"""A stand-in for a game (or an encoder) on the same GPU: keeps the card busy for a set share of the time.

usage: python -m bench.gpu_load --duty 0.7 [--period-ms 16.7] [--vram-gb 0] [--secs 0] [--after 0]

Every ``period`` (one 60 fps frame by default) it runs matrix products for ``duty`` x period and then sleeps, like
a game that renders a frame and waits for the next one. Two processes on one card take turns, so the engine's
models slow down the way they do beside a game. ``--vram-gb`` also holds video memory, ``--after`` starts the load
that many seconds late (a game started in mid-stream). It prints the share of time it really got every 5 s.
Needs PyTorch with CUDA (bench only; the product never imports this).
"""

from __future__ import annotations

import argparse
import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench.gpu_load")
    ap.add_argument("--duty", type=float, required=True, help="share of every period the GPU is kept busy, 0..1")
    ap.add_argument("--period-ms", type=float, default=1000 / 60)
    ap.add_argument("--size", type=int, default=1536, help="side of the square matrices (one product = a few ms)")
    ap.add_argument("--vram-gb", type=float, default=0.0)
    ap.add_argument("--secs", type=float, default=0.0, help="stop after this long; 0 = until killed")
    ap.add_argument("--after", type=float, default=0.0, help="start the load this many seconds late")
    args = ap.parse_args(argv)
    import torch

    if args.after > 0:
        time.sleep(args.after)
    held = torch.empty(int(args.vram_gb * 2**30), dtype=torch.uint8, device="cuda") if args.vram_gb > 0 else None
    x = torch.randn(args.size, args.size, device="cuda", dtype=torch.half)
    period = args.period_ms / 1000
    start = last = time.perf_counter()
    busy = 0.0
    print(
        f"GPU_LOAD start duty {args.duty} period {args.period_ms:.1f} ms vram {0 if held is None else held.numel() >> 20} MiB", flush=True
    )
    while args.secs <= 0 or time.perf_counter() - start < args.secs:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.duty * period:
            (x @ x).sum()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        busy += t1 - t0
        rest = t0 + period - t1
        if rest > 0:
            time.sleep(rest)
        if t1 - last >= 5:
            print(f"GPU_LOAD busy {busy / (t1 - last):.2f}", flush=True)
            last, busy = time.perf_counter(), 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
