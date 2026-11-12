"""Entry point of the frozen Windows build (StreamSafe.exe).

StreamSafe.exe check [--json]      system check (GPU, runtime, FFmpeg, OBS, models)
StreamSafe.exe prepare --models D  build/verify the per-PC engines (NVIDIA TensorRT)
StreamSafe.exe run --input ... --output ...   the stream filter (same options as python -m streamsafe)
"""

import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()  # the analyzer runs in a child process; a frozen exe re-enters here
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "check":
        from streamsafe.syscheck import main as m
    elif cmd == "prepare":
        from streamsafe.prepare import main as m
    elif cmd == "run":
        from streamsafe.cli import main as m
    else:
        print(__doc__)
        return 0 if cmd in ("", "-h", "--help") else 2
    return m(rest)


if __name__ == "__main__":
    sys.exit(main())
