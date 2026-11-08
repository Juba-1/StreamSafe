"""Why a modest person is covered: per decision, the biggest person's risk score, the beach prior and the
threshold (diag JSON from --diag). usage: python -m bench.lucia_probe DIAG.json [SECONDS=12] [FPS=30]"""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> None:
    with open(argv[0]) as f:
        diag = json.load(f)
    secs = float(argv[1]) if len(argv) > 1 else 12.0
    fps = float(argv[2]) if len(argv) > 2 else 30.0
    for fid, dets, *extra in diag["dec"]:
        if fid > secs * fps or not dets:
            continue
        big = max(dets, key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]))
        tid, box, p, risky, _stale = big
        p_swim, thr = extra if extra else (None, None)
        print(f"t={fid / fps:5.2f} tid={tid} h={box[3] - box[1]:4d} p={p:.3f} risky={risky} p_swim={p_swim} thr={thr}")


if __name__ == "__main__":
    main(sys.argv[1:])
