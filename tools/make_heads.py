"""Put the risk heads into ``OUT/heads`` for the installer.

    python tools/make_heads.py OUT_DIR [--require-real]

Real heads: ``installer/heads/`` (fast_h6.txt, male_h6.txt, thr_h6.txt, optional scene_h.txt, heads.json). Without
them a synthetic PLACEHOLDER set is written (so a CI test build can be assembled and smoke-tested); the app refuses to
stream with it. ``--require-real`` (release builds) makes missing real heads an error.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamsafe.app import heads  # noqa: E402


def placeholder(folder: Path) -> None:
    n = 16
    line = f"1000;0;{','.join(['1'] * n)}"
    (folder / "fast_h6.txt").write_text(line, encoding="utf-8")
    (folder / "male_h6.txt").write_text(line, encoding="utf-8")
    (folder / "thr_h6.txt").write_text("0.5", encoding="utf-8")
    (folder / heads.META).write_text(json.dumps({"version": 0}), encoding="utf-8")
    (folder / heads.PLACEHOLDER).write_text(
        "CI test build: these heads decide nothing real. The app does not stream with them.\n", encoding="utf-8"
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    out = Path(argv[0]) / "heads"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    src = ROOT / "installer" / "heads"
    real = all((src / name).is_file() for name in heads.FILES)
    if real:
        for name in (*heads.FILES, *heads.OPTIONAL, heads.META):
            if (src / name).is_file():
                shutil.copyfile(src / name, out / name)
        if not (out / heads.META).is_file():
            (out / heads.META).write_text(json.dumps({"version": 1}), encoding="utf-8")
        if why := heads.problem(out):
            print(f"the heads in installer/heads do not load: {why}", file=sys.stderr)
            return 1
        print(f"real heads, version {heads.version(out)}")
        return 0
    if "--require-real" in argv:
        print("installer/heads has no real heads (fast_h6.txt, male_h6.txt, thr_h6.txt): a release build needs them", file=sys.stderr)
        return 1
    placeholder(out)
    print("PLACEHOLDER heads written (test build): the app will not stream with them")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
