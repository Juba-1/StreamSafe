"""Draw the eye's score for every person on the original frames (from a --diag log): red = covered, green = not,
with p (person score) and the threshold. Usage: annotate_diag.py VIDEO DIAG.json OUT.jpg [every_s]"""

import json
import sys

import cv2
import numpy as np


def main() -> None:
    video, diag, out = sys.argv[1:4]
    every = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    with open(diag, encoding="utf-8") as f:
        data = json.load(f)
    decisions = data.get("dec", data) if isinstance(data, dict) else data
    by_frame = {int(d[0]): d for d in decisions if isinstance(d, list) and len(d) >= 4}
    frames = sorted(by_frame)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tiles, t, n = [], 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while t * fps < n and len(tiles) < 48:
        idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, im = cap.read()
        t += every
        if not ok:
            continue
        im = cv2.resize(im, (1920, 1080))
        near = min(frames, key=lambda f: abs(f - idx)) if frames else None
        if near is not None and abs(near - idx) <= 6:
            _, dets, _, thr = by_frame[near][:4]
            for _tid, box, p, risky, _stale in dets:
                x0, y0, x1, y1 = map(int, box)
                col = (40, 40, 255) if risky else (60, 220, 60)
                cv2.rectangle(im, (x0, y0), (x1, y1), col, 4)
                cv2.putText(im, f"{p:.2f}", (x0 + 4, max(30, y0 + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4)
            cv2.putText(im, f"thr {thr:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
        tiles.append(cv2.resize(im, (480, 270)))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("SCORES", out, len(tiles))


if __name__ == "__main__":
    main()
