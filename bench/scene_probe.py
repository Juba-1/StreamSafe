"""Run 41: what the scene model and the HUD reader say on real clips, before trusting the scene layer.

For every clip, frames at FPS (default 2): explicit / beach probabilities and the best prompt; with HUD=1
also the OCR text and keyword. Writes OUT/scene_probe.csv, per-clip summaries, and contact sheets of the
highest-scoring frames of every clip (OUT/sheets/<clip>.jpg, 320x180 thumbs with p) for visual review,
and OUT/scene_emb.npz (clip, t, emb fp16) so a scene head can be trained on labeled frames later.

usage: python -m bench.scene_probe OUT_DIR VIDEO [VIDEO ...]      env: FPS=2 SECS=600 HUD=0 TOP=24
Prints "SCENE_CLIP <json>" per clip and "SCENE_TIMING <json>".
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from streamsafe.analysis.scene import hud_hit
from streamsafe.io.sources import VideoFileSource
from streamsafe.models.torch_backend.scene import EXPLICIT_PROMPTS, OTHER_PROMPTS, SWIM_PROMPTS, SigLIPScene

PROMPTS = SWIM_PROMPTS + EXPLICIT_PROMPTS + OTHER_PROMPTS
THUMB = (320, 180)


def sheet(rows: list[tuple[float, np.ndarray, str]], path: Path) -> None:
    cells = []
    for p, thumb, note in rows:
        t = thumb.copy()
        cv2.rectangle(t, (0, 0), (THUMB[0], 22), (0, 0, 0), -1)
        cv2.putText(t, f"{p:.2f} {note}"[:40], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cells.append(t)
    while len(cells) % 6:
        cells.append(np.zeros((THUMB[1], THUMB[0], 3), np.uint8))
    grid = np.vstack([np.hstack(cells[i : i + 6]) for i in range(0, len(cells), 6)])
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    fps, secs = float(os.environ.get("FPS", 2)), float(os.environ.get("SECS", 600))
    top = int(os.environ.get("TOP", 24))
    hud = None
    if os.environ.get("HUD") == "1":
        from streamsafe.models.ocr_hud import RapidHud

        hud = RapidHud()
    scene = SigLIPScene()
    t_scene, t_hud = [], []
    all_emb, all_key = [], []
    with open(out / "scene_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "t", "p_explicit", "p_swim", "best_prompt", "hud_text", "hud_hit"])
        for path in argv[1:]:
            name = Path(path).stem
            src = VideoFileSource(path, fps, secs)
            rows, pe = [], []
            for k, frame in enumerate(src.frames()):
                rgb = cv2.cvtColor(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                s = time.perf_counter()
                emb, p = scene.embed_probs(rgb)
                all_emb.append(emb.astype(np.float16))
                all_key.append((name, round(k / fps, 2)))
                t_scene.append(1000 * (time.perf_counter() - s))
                p_e = float(p[len(SWIM_PROMPTS) : len(SWIM_PROMPTS) + len(EXPLICIT_PROMPTS)].sum())
                text = hit = ""
                if hud is not None:
                    s = time.perf_counter()
                    text = hud(frame)
                    t_hud.append(1000 * (time.perf_counter() - s))
                    hit = hud_hit(text)
                best = PROMPTS[int(np.argmax(p))].replace("a video game screenshot of ", "")
                w.writerow([name, round(k / fps, 2), round(p_e, 4), round(float(p[: len(SWIM_PROMPTS)].sum()), 4), best, text[:120], hit])
                pe.append(p_e)
                note = ("HUD " if hit else "") + f"{k / fps:.0f}s {best}"
                rows.append((p_e, cv2.resize(frame, THUMB, interpolation=cv2.INTER_AREA), note))
            src.close()
            if not pe:
                continue
            a = np.array(pe)
            rep = dict(
                clip=name,
                n=len(a),
                mean=round(float(a.mean()), 3),
                p95=round(float(np.percentile(a, 95)), 3),
                max=round(float(a.max()), 3),
                **{f"share_ge_{t}": round(float((a >= t).mean()), 3) for t in (0.3, 0.4, 0.5, 0.6, 0.7)},
            )
            print("SCENE_CLIP " + json.dumps(rep), flush=True)
            rows.sort(key=lambda r: -r[0])
            sheet(rows[:top], out / "sheets" / f"{name}.jpg")
    if all_emb:
        clips, times = zip(*all_key, strict=True)
        np.savez_compressed(out / "scene_emb.npz", clip=np.array(clips), t=np.array(times), emb=np.stack(all_emb))
    timing = dict(scene_ms_p50=round(float(np.median(t_scene)), 2), scene_ms_p95=round(float(np.percentile(t_scene, 95)), 2))
    if t_hud:
        timing |= dict(hud_ms_p50=round(float(np.median(t_hud)), 2), hud_ms_p95=round(float(np.percentile(t_hud, 95)), 2))
    print("SCENE_TIMING " + json.dumps(timing), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
