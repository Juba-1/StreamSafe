"""Command line: ``python -m streamsafe --input IN --output OUT [options]``.

Inputs : a video file | ``srt://...`` / ``rtmp://...`` / ``udp://...`` (any FFmpeg input) | ``device:<name|index>``
Outputs: an .mp4 review file | ``rtmp(s)://...`` / ``srt://...`` (H.264 via FFmpeg) | ``window`` (repeatable)
Every engine setting can also be set from the environment (see ``streamsafe/config.py``).

Examples
  # benchmark a file deterministically
  python -m streamsafe --input trailer.mp4 --output review.mp4 --mode offline
  # OBS streams to us over SRT, we stream the protected picture to Kick (with its sound, delayed alike and
  # silent while the game audio is muted)
  python -m streamsafe --input "srt://127.0.0.1:9000?mode=listener" --output "rtmps://<ingest>/app/<key>"
  # OBS Virtual Camera (output type "Source" = the game capture) in, a window for OBS to capture out
  python -m streamsafe --input "device:OBS Virtual Camera" --output window
"""

from __future__ import annotations

import argparse
import sys

from streamsafe import session_report
from streamsafe.config import Config
from streamsafe.io.sinks import MultiSink, open_sink
from streamsafe.io.sources import open_source
from streamsafe.runtime.runner import run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="streamsafe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="video file, FFmpeg URL or device:<name|index>")
    p.add_argument("--output", action="append", default=[], help="mp4 file, FFmpeg URL or 'window' (repeatable)")
    p.add_argument("--mode", choices=["realtime", "offline"], help="default: realtime")
    p.add_argument("--delay", type=float, help="stream delay in seconds")
    p.add_argument("--secs", type=float, help="stop a file input after this many seconds")
    p.add_argument("--size", help="output frame size WxH (default: the input's own 16:9 size)")
    p.add_argument("--head-dir", help="folder with the eye's risk heads")
    p.add_argument("--stats", help="write the run summary JSON here")
    p.add_argument("--diag", help="write every decision and every drawn garment here (JSON)")
    p.add_argument("--report-dir", help="folder for the tester session report (default: %%LOCALAPPDATA%%/StreamSafe/reports)")
    p.add_argument("--no-report", action="store_true", help="do not write the tester session report")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = dict(mode=args.mode, delay=args.delay, secs=args.secs, head_dir=args.head_dir, stats_out=args.stats, diag_out=args.diag)
    return Config.from_env(**{k: v for k, v in overrides.items() if v is not None})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    source = open_source(args.input, cfg.runtime.fps, cfg.runtime.secs, size, cfg.output.audio)  # type: ignore[arg-type]
    sinks = [open_sink(spec, source.size, cfg.runtime.fps, audio=cfg.output.audio) for spec in args.output]
    sink = MultiSink(sinks) if len(sinks) > 1 else (sinks[0] if sinks else None)
    try:
        report = run(cfg, source, sink)
    except KeyboardInterrupt:  # interrupted outside the frame loop (start-up or shutdown): nothing to report
        return 130
    finally:
        source.close()
        if sink is not None:
            sink.close()
    if not args.no_report:
        try:
            print("SESSION_REPORT", session_report.write(report, cfg.runtime.fps, args.report_dir), flush=True)
        except OSError as exc:  # a report must never break a stream that already ended
            print("SESSION_REPORT_FAILED", exc, flush=True)
    return 1 if report["summary"]["analyzer_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
