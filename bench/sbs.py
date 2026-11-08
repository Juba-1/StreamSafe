"""Side-by-side review video: OLD (left) | NEW (right), 960x540 each -> 1920x540 H.264.
usage: python -m bench.sbs OLD.mp4 NEW.mp4 OUT.mp4"""

import sys

import cv2
import imageio_ffmpeg
import numpy as np


def label(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(frame, (0, 0), (110, 44), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return frame


def main(old: str, new: str, out: str) -> None:
    a, b = cv2.VideoCapture(old), cv2.VideoCapture(new)
    writer = imageio_ffmpeg.write_frames(out, (1920, 540), fps=30, codec="libx264", quality=7, macro_block_size=8, ffmpeg_log_level="error")
    writer.send(None)
    while True:
        ok_a, fa = a.read()
        ok_b, fb = b.read()
        if not (ok_a and ok_b):
            break
        pair = np.hstack([label(cv2.resize(fa, (960, 540)), "OLD"), label(cv2.resize(fb, (960, 540)), "NEW")])
        writer.send(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB))
    writer.close()
    print("SBS", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
