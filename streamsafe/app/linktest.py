""" "Test connection": is the platform taking a stream at this address with this key?

Sends a few seconds of the black slate with silence - nothing of the game, nothing of the camera - and says
whether the platform accepted it. The channel IS live for those seconds (a platform has no way to check a key
without a stream), which the window says before it starts. Output: ``STATUS`` lines (``io/link.py``) and one last
line ``LINK_TEST ok`` / ``LINK_TEST failed <reason>``. The address comes from the environment, never from the
command line, and is never printed.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from streamsafe.io import link
from streamsafe.io.sinks import FFmpegSink, PlatformRefused

OK_EXIT, FAILED_EXIT = 0, 8
WAIT_S = 30.0  # for the platform to take the stream


def run(url: str, seconds: float = 4.0, size: tuple[int, int] = (1280, 720), fps: float = 30.0, sink: FFmpegSink | None = None) -> str:
    """ "" when the platform took the stream for ``seconds``, else the reason (a word of ``io/link.py``)."""
    sink = sink or FFmpegSink(url, size, fps)
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    start = time.monotonic()
    up_since: float | None = None
    try:
        while True:
            now = time.monotonic()
            sink.write(frame, None) if sink.takes_audio else sink.write(frame)
            if sink.connected and up_since is None:
                up_since = now
            if not sink.connected:
                up_since = None
            if up_since is not None and now - up_since >= seconds:
                return ""
            if now - start > WAIT_S + seconds:
                return "timeout"
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - now)))
    except PlatformRefused as exc:
        return exc.reason
    finally:
        sink.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="link-test", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="the platform address (give it as env:NAME)")
    p.add_argument("--seconds", type=float, default=4.0)
    args = p.parse_args(argv)
    if not args.output.startswith(("rtmp://", "rtmps://")):
        print("LINK_TEST failed bad_rtmp", flush=True)
        return FAILED_EXIT
    try:
        why = run(args.output, args.seconds)
    except Exception as exc:
        why = link.Secrets(args.output).clean(f"error {type(exc).__name__}: {exc}")
    print("LINK_TEST ok" if not why else f"LINK_TEST failed {why}", flush=True)
    return OK_EXIT if not why else FAILED_EXIT
