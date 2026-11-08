"""Where two head sets disagree on the same video: a contact sheet of the people one covers and the other does not.

Both runs must be offline runs of the same clip (same decisions, same detector); people are paired by frame and box.
One cell per track and side (the frame where the two scores differ most), at most 30 per side.

usage: python bench/disagree.py VIDEO NAME_A=diag_a.json NAME_B=diag_b.json OUT.jpg
Prints "DISAGREE <json>" (person-decisions covered by only A / only B, tracks) and writes the sheet:
left block = covered only by A, right block = covered only by B; each cell shows the person in context.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from streamsafe.types import ANALYSIS_SIZE

PER_SIDE = 30
CELL_W, CELL_H = 160, 240
SAME_BOX = 0.8


def _iou(a: list[int], b: list[int]) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def pairs(dec_a: list, dec_b: list) -> list[tuple[int, list[int], float, float, int, int, int]]:
    """(frame, box, p_a, p_b, risky_a, risky_b, track id in A) for every person seen by both runs."""
    b_by_frame = {d[0]: d[1] for d in dec_b}
    out = []
    for fid, people, *_ in dec_a:
        others = b_by_frame.get(fid, [])
        for tid, box, p, risky, _stale in people:
            best = max(others, key=lambda o: _iou(box, o[1]), default=None)
            if best is not None and _iou(box, best[1]) >= SAME_BOX:
                out.append((fid, box, p, best[2], int(risky), int(best[3]), tid))
    return out


def _cell(frame: np.ndarray, box: list[int], text: str) -> np.ndarray:
    sx, sy = frame.shape[1] / ANALYSIS_SIZE[0], frame.shape[0] / ANALYSIS_SIZE[1]
    x1, y1, x2, y2 = (round(v * s) for v, s in zip(box, (sx, sy, sx, sy), strict=True))
    pad_x, pad_y = (x2 - x1) // 2 + 10, (y2 - y1) // 4 + 10
    h, w = frame.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
    view = frame[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(view, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 255), 2)
    k = min(CELL_W / view.shape[1], (CELL_H - 20) / view.shape[0])
    view = cv2.resize(view, (max(1, int(view.shape[1] * k)), max(1, int(view.shape[0] * k))))
    cell = np.zeros((CELL_H, CELL_W, 3), np.uint8)
    cell[20 : 20 + view.shape[0], : view.shape[1]] = view
    cv2.putText(cell, text, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return cell


def main(argv: list[str]) -> None:
    video, out = argv[0], argv[-1]
    (name_a, path_a), (name_b, path_b) = (a.split("=", 1) for a in argv[1:3])
    with open(path_a) as fa, open(path_b) as fb:
        rows = pairs(json.load(fa)["dec"], json.load(fb)["dec"])
    only = {name_a: [r for r in rows if r[4] and not r[5]], name_b: [r for r in rows if r[5] and not r[4]]}
    picks: dict[str, list] = {}
    for side, items in only.items():
        best: dict[int, tuple] = {}
        for r in items:  # one per track: the largest score gap
            if r[6] not in best or abs(r[2] - r[3]) > abs(best[r[6]][2] - best[r[6]][3]):
                best[r[6]] = r
        picks[side] = sorted(best.values(), key=lambda r: -abs(r[2] - r[3]))[:PER_SIDE]
    stats = {
        "pairs": len(rows),
        "covered_" + name_a: sum(r[4] for r in rows),
        "covered_" + name_b: sum(r[5] for r in rows),
        "only_" + name_a: len(only[name_a]),
        "only_" + name_b: len(only[name_b]),
        "tracks_only_" + name_a: len({r[6] for r in only[name_a]}),
        "tracks_only_" + name_b: len({r[6] for r in only[name_b]}),
    }
    print("DISAGREE " + json.dumps(stats), flush=True)
    cap = cv2.VideoCapture(video)
    blocks = []
    for side, items in picks.items():
        cells = []
        for fid, box, pa, pb, *_ in sorted(items):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if ok:
                cells.append(_cell(frame, box, f"{side} f{fid} {pa:.2f}/{pb:.2f}"))
        cells += [np.zeros((CELL_H, CELL_W, 3), np.uint8)] * (-len(cells) % 6 or (6 if not cells else 0))
        grid = np.vstack([np.hstack(cells[i : i + 6]) for i in range(0, len(cells), 6)])
        title = np.zeros((30, grid.shape[1], 3), np.uint8)
        cv2.putText(title, f"covered only by {side}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        blocks.append(np.vstack([title, grid]))
    height = max(b.shape[0] for b in blocks)
    blocks = [np.vstack([b, np.zeros((height - b.shape[0], b.shape[1], 3), np.uint8)]) for b in blocks]
    sep = np.full((height, 8, 3), 255, np.uint8)
    cv2.imwrite(out, np.hstack([blocks[0], sep, blocks[1]]), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main(sys.argv[1:])
