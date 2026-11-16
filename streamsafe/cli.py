"""Command line: ``python -m streamsafe --input IN --output OUT [options]``.

Inputs : a video file (``file:<path>``: with its sound) | ``srt://...`` / ``rtmp://...`` / ``udp://...`` (any FFmpeg
         input) | ``device:<name|index>`` (``device:<name>|<sound device>``: a capture card with its sound)
Outputs: an .mp4 review file | ``rtmp(s)://...`` / ``srt://...`` (H.264 via FFmpeg) | ``record:<file.mp4>`` (the
         protected picture and sound in full quality) | ``window`` (repeatable)
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
from collections.abc import Callable

from streamsafe import session_report
from streamsafe.config import Config
from streamsafe.io import link
from streamsafe.io.sinks import MultiSink, PlatformRefused, open_sink
from streamsafe.io.sources import open_source
from streamsafe.runtime.runner import AnalyzerFactory, run


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


INPUT_FAILED_EXIT = 7  # the input could not be opened (the SRT port is busy, no such device)
PLATFORM_FAILED_EXIT = 8  # the platform never accepted the stream (wrong address or key, no network)


class Refused(Exception):  # noqa: N818 - a refusal, not an error
    """A check between opening the input and opening the outputs said no (``before_outputs``): ``code`` is the
    exit code, ``reason`` a short word for the report."""

    def __init__(self, code: int, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.code, self.reason = code, reason


def main(
    argv: list[str] | None = None,
    factory: AnalyzerFactory | None = None,
    before_outputs: Callable[[list[str]], None] | None = None,
) -> int:
    """``factory``: another analyzer than the product's (tests, ``selftest-stream``). ``before_outputs``: called
    with the output specs once the input is open and before any output is - the installed program does its slower
    start checks there (graphics card, encoder), so the SRT listener is already up while they run."""
    args = parse_args(argv)
    cfg = config_from_args(args)
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    secrets = [link.Secrets(spec) for spec in args.output]
    source, sink, report = None, None, None
    code, reason, detail = 0, "", ""
    try:
        failed = True
        try:
            # the input first and at once: OBS can connect from this moment on
            source = open_source(args.input, cfg.runtime.fps, cfg.runtime.secs, size, cfg.output.audio)  # type: ignore[arg-type]
            if before_outputs is not None:
                before_outputs(args.output)
            sinks = [open_sink(spec, source.size, cfg.runtime.fps, audio=cfg.output.audio) for spec in args.output]
            sink = MultiSink(sinks) if len(sinks) > 1 else (sinks[0] if sinks else None)
            report = run(cfg, source, sink, factory) if factory is not None else run(cfg, source, sink)
            failed = False
        finally:
            for part in (source, sink):
                try:
                    if part is not None:
                        part.close()
                except Exception:  # a sink says again at close why it broke: only news when the run itself went well
                    if not failed:
                        raise
    except KeyboardInterrupt:  # stopped during start-up or shutdown: still leave a report saying so
        report = report or _not_started_report(stopped_by_user=True)
        code, reason = 130, "user"
    except Refused as exc:
        print(str(exc), flush=True)
        code, reason, detail = exc.code, exc.reason, str(exc)
    except PlatformRefused as exc:
        code, reason, detail = PLATFORM_FAILED_EXIT, "platform_failed", f"{exc.reason}: {exc}"
    except Exception as exc:  # could not open the input / output, or the run broke: the report says so, then the error
        text = f"{type(exc).__name__}: {exc}"
        for s in secrets:
            text = s.clean(text)
        print(f"RUN_FAILED {text}", flush=True)
        _finish(args, report, cfg.runtime.fps, "error", text, source, sink)
        raise
    else:
        code = 1 if report["summary"]["analyzer_errors"] else 0
        if report["summary"].get("stopped_by_user"):
            reason = "user"
        elif getattr(source, "error", ""):
            code, reason, detail = INPUT_FAILED_EXIT, "input_failed", source.error
        else:
            reason = "input_ended"
    _finish(args, report, cfg.runtime.fps, reason, detail, source, sink)
    return code


def _finish(args: argparse.Namespace, report: dict | None, fps: float, reason: str, detail: str, source: object, sink: object) -> None:
    """The last words of a run: why it ended (``STATUS stopped``) and the session report with that reason."""
    report = report or _not_started_report(stopped_by_user=reason == "user")
    summary = report.setdefault("summary", {})
    summary["stop_reason"], summary["stop_detail"] = reason, detail[:400]
    summary["input_connections"] = getattr(source, "connections", None)
    inner = [getattr(s, "sink", s) for s in getattr(sink, "sinks", [sink] if sink is not None else [])]
    inner = [getattr(s, "sink", s) for s in inner]
    summary["platform_reconnects"] = sum(getattr(s, "reconnects", 0) for s in inner)
    summary["unsent_frames"] = sum(getattr(s, "unsent", 0) for s in inner)
    link.status("stopped", reason=reason, detail=detail[:400])
    _write_report(args, report, fps)


def _not_started_report(stopped_by_user: bool) -> dict:
    return {"summary": {"frames_out": 0, "seconds": 0, "analyzer_errors": 0, "stopped_by_user": stopped_by_user}}


def _write_report(args: argparse.Namespace, report: dict, fps: float) -> None:
    if not args.no_report:
        try:
            print("SESSION_REPORT", session_report.write(report, fps, args.report_dir), flush=True)
        except OSError as exc:  # a report must never break a stream that already ended
            print("SESSION_REPORT_FAILED", exc, flush=True)


if __name__ == "__main__":
    sys.exit(main())
