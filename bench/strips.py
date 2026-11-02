"""Frame strips from a side-by-side video: at each time, 6 frames 0.1 s apart, stacked -> one jpg.
usage: python bench/strips.py SBS.mp4 OUT_PREFIX 1.0,6.7,21.8"""

import sys

import cv2
import numpy as np


def main(video: str, prefix: str, times: list[float]) -> None:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    for t in times:
        rows = []
        for k in range(6):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round((t + 0.1 * k) * fps))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (1440, 408))
            cv2.putText(frame, f"{t + 0.1 * k:.1f}s", (680, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            rows.append(frame)
        if rows:
            cv2.imwrite(f"{prefix}_{t:.1f}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
            print("strip", prefix, t, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [float(t) for t in sys.argv[3].split(",")])
