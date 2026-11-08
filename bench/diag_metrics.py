"""Cover-quality metrics from an engine diag log (``--diag``) and an independent per-frame person
detection of the same source frames.

usage: python -m bench.diag_metrics SRC.mp4 SECS LABEL=diag.json [LABEL=diag.json ...]

lag        garment centre vs the detected person's centre in the SAME shown frame (fraction of width / height)
off_body   share of the garment box outside the person box (trailing garment / ghost)
ghosts     frames where one person gets 2+ garments that do not overlap each other (double cover)
size_jump  frame-to-frame garment height jumps > 15 % on the same person (per 10 s)
torso      share of a covered person's torso band (x 30-70 %, y 20-75 % of the box) inside garment boxes;
           p10 and the share of frames under 80 % = visibly exposed while "covered"
jitter     per garment track: |c(t+1) - 2 c(t) + c(t-1)| / garment height (c = garment centre, 3 consecutive
           frames) - shaking; the same for the detected person box (det_jitter) is the true motion + detector noise
pops       frames where that acceleration is > 4 % of the height (a visible jump), per 10 s; switch_pop_p90 =
           the acceleration on frames where the garment changes to another decision's shape
early      per track in the first 25 s: first second, risky decisions, decisions, max p, height
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

Box = tuple[int, int, int, int]
FPS = 30.0
MIN_H = 36


def detect_people(src: str, secs: float) -> list[list[Box]]:
    """Person boxes per output frame (same 30 fps resampling as the engine)."""
    import rfdetr

    det = rfdetr.RFDETRSegNano()
    with contextlib.suppress(Exception):  # older rfdetr: stays fp32
        det.optimize_for_inference(compile=False, dtype=torch.float16)
    cap = cv2.VideoCapture(src)
    step = (cap.get(cv2.CAP_PROP_FPS) or FPS) / FPS
    boxes: list[list[Box]] = []
    read, pos, frame = 0, 0.0, None
    while len(boxes) < int(secs * FPS):
        while read <= int(pos):
            ok, frame = cap.read()
            read += 1
            if not ok:
                return boxes
        im = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        d = det.predict(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), threshold=0.35)
        cls = np.asarray(d.class_id) if d.class_id is not None else np.zeros(0)
        keep = (cls == 1) if (cls == 1).any() else (cls == 0)
        boxes.append([tuple(int(v) for v in d.xyxy[i]) for i in np.where(keep)[0] if d.xyxy[i][3] - d.xyxy[i][1] >= MIN_H])
        pos += step
    return boxes


def inter(a: Box, b: Box) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(b: Box) -> float:
    return max(1, (b[2] - b[0]) * (b[3] - b[1]))


def iou(a: Box, b: Box) -> float:
    return inter(a, b) / (area(a) + area(b) - inter(a, b))


def torso_cover(person: Box, garments: list[Box]) -> float:
    pw, ph = person[2] - person[0], person[3] - person[1]
    bx1, bx2, by1, by2 = person[0] + 0.3 * pw, person[0] + 0.7 * pw, person[1] + 0.2 * ph, person[1] + 0.75 * ph
    gw, gh = max(1, int(bx2 - bx1)), max(1, int(by2 - by1))
    band = np.zeros((gh, gw), bool)
    for g in garments:
        a1, a2 = int(max(0, g[0] - bx1)), int(min(gw, g[2] - bx1))
        c1, c2 = int(max(0, g[1] - by1)), int(min(gh, g[3] - by1))
        if a2 > a1 and c2 > c1:
            band[c1:c2, a1:a2] = True
    return float(band.mean())


POP = 0.04  # acceleration (fraction of height) that reads as a visible jump


def jitter(frm: list, people: list[list[Box]]) -> dict:
    """Shaking of each garment (and of its detected person, for reference) from 3 consecutive frames."""
    series: dict[int, dict[int, tuple]] = defaultdict(dict)
    for v, tid, kf, y, x, h, w, _used, _col in frm:
        series[tid].setdefault(v, (y + h / 2, x + w / 2, h, kf, (x, y, x + w, y + h)))
    acc, det_acc, switch = [], [], []
    for track in series.values():
        for v, (cy, cx, h, kf, box) in track.items():
            a, b = track.get(v - 1), track.get(v + 1)
            if a is None or b is None:
                continue
            ay = abs(b[0] - 2 * cy + a[0]) / max(1, h)
            ax = abs(b[1] - 2 * cx + a[1]) / max(1, h)
            acc.append(max(ay, ax))
            if a[3] != kf or b[3] != kf:
                switch.append(acc[-1])
            pb = [_match(people, u, g[4]) for u, g in ((v - 1, a), (v, (cy, cx, h, kf, box)), (v + 1, b))]
            if all(p is not None for p in pb):
                c = [((p[1] + p[3]) / 2, (p[0] + p[2]) / 2) for p in pb]
                ph = max(1, pb[1][3] - pb[1][1])
                det_acc.append(max(abs(c[2][0] - 2 * c[1][0] + c[0][0]), abs(c[2][1] - 2 * c[1][1] + c[0][1])) / ph)
    seconds = max(1, len(people)) / FPS

    def q(values: list, p: float) -> float | None:
        return round(float(np.percentile(values, p)), 4) if values else None

    return dict(
        jitter_p50=q(acc, 50),
        jitter_p90=q(acc, 90),
        jitter_p99=q(acc, 99),
        pops_per10s=round(10 * sum(a > POP for a in acc) / seconds, 2),
        switch_pop_p90=q(switch, 90),
        det_jitter_p50=q(det_acc, 50),
        det_jitter_p90=q(det_acc, 90),
        det_pops_per10s=round(10 * sum(a > POP for a in det_acc) / seconds, 2),
    )


def _match(people: list[list[Box]], v: int, g: Box) -> Box | None:
    if v >= len(people) or not people[v]:
        return None
    pb = max(people[v], key=lambda p: inter(g, p) / area(g))
    return pb if inter(g, pb) / area(g) >= 0.2 else None


def metrics(diag: dict, people: list[list[Box]]) -> dict:
    drawn: dict[int, list] = defaultdict(list)
    for v, _tid, _kf, y, x, h, w, used, _col in diag["frm"]:
        drawn[v].append(((x, y, x + w, y + h), used))
    lag_x, lag_y, off_body, torso = [], [], [], []
    ghosts = size_jumps = n_layers = n_motion = 0
    last: dict = {}  # person key -> (frame, box, garment height)
    for v, layers in drawn.items():
        if v >= len(people):
            continue
        per_person: dict[int, list[Box]] = defaultdict(list)
        for g, used in layers:
            n_layers += 1
            n_motion += used
            if not people[v]:
                continue
            j = max(range(len(people[v])), key=lambda k: inter(g, people[v][k]) / area(g))
            pb = people[v][j]
            overlap = inter(g, pb) / area(g)
            if overlap < 0.2:
                continue  # garment of someone the per-frame detector does not see
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            lag_x.append(abs((g[0] + g[2]) / 2 - (pb[0] + pb[2]) / 2) / max(1, pw))
            lag_y.append(abs((g[1] + g[3]) / 2 - (pb[1] + pb[3]) / 2) / max(1, ph))
            off_body.append(1 - overlap)
            per_person[j].append(g)
        all_garments = [g for g, _ in layers]
        for j, garments in per_person.items():
            pb = people[v][j]
            torso.append(torso_cover(pb, all_garments))
            pairs = [(a, b) for i, a in enumerate(garments) for b in garments[i + 1 :]]
            if any(inter(a, b) / min(area(a), area(b)) < 0.5 for a, b in pairs):
                ghosts += 1
            key = next((k for k, (pv, ppb, _) in last.items() if pv == v - 1 and iou(ppb, pb) > 0.5), None)
            height = max(g[3] - g[1] for g in garments)
            if key is not None and abs(height - last[key][2]) > 0.15 * max(height, 1):
                size_jumps += 1
            last[key if key is not None else (v, j)] = (v, pb, height)

    seconds = max(1, len(people)) / FPS

    def q(values: list, p: float) -> float | None:
        return round(float(np.percentile(values, p)), 3) if values else None

    early: dict[int, dict] = {}
    for fid, dets, *_ in diag["dec"]:
        if fid > 25 * FPS:
            break
        for tid, box, p, risky, _stale in dets:
            e = early.setdefault(tid, dict(first=fid, risky_dec=0, dec=0, maxp=0.0, h=box[3] - box[1]))
            e["dec"] += 1
            e["risky_dec"] += risky
            e["maxp"] = max(e["maxp"], p)
    early_risky = sorted(
        (
            [k, round(e["first"] / FPS, 1), e["risky_dec"], e["dec"], round(e["maxp"], 2), e["h"]]
            for k, e in early.items()
            if e["risky_dec"]
        ),
        key=lambda z: z[1],
    )[:25]
    return dict(
        late_cover=late_cover(diag["dec"]),
        layer_frames=n_layers,
        motion_used=round(n_motion / max(1, n_layers), 3),
        lag_x_p50=q(lag_x, 50),
        lag_x_p90=q(lag_x, 90),
        lag_y_p90=q(lag_y, 90),
        off_body_p50=q(off_body, 50),
        off_body_p90=q(off_body, 90),
        far_off_share=round(float(np.mean(np.array(lag_x) > 0.25)), 4) if lag_x else None,
        ghosts_per10s=round(10 * ghosts / seconds, 2),
        torso_p10=q(torso, 10),
        torso_under80=round(float(np.mean(np.array(torso) < 0.8)), 4) if torso else None,
        size_jump_per10s=round(10 * size_jumps / seconds, 2),
        **jitter(diag["frm"], people),
        early_risky_tracks=early_risky,
    )


DELAY_S = 2.0  # the engine's default delay: a late cover inside it can be moved back before viewers see it


def late_cover(decisions: list) -> dict:
    """How late tracks that end up covered were covered: frames between a track's first decision and its
    first risky decision. All of it was shown uncovered; the part within DELAY_S could be covered by
    looking ahead in the delay buffer."""
    first: dict[int, int] = {}
    first_risky: dict[int, int] = {}
    for fid, dets, *_ in decisions:
        for tid, _box, _p, risky, _stale in dets:
            first.setdefault(tid, fid)
            if risky:
                first_risky.setdefault(tid, fid)
    lags = [(first_risky[t] - first[t]) / FPS for t in first_risky]
    late = [s for s in lags if s > 0]
    return dict(
        covered_tracks=len(lags),
        late_tracks=len(late),
        late_s_total=round(sum(late), 2),
        late_s_fixable=round(sum(min(s, DELAY_S) for s in late), 2),
        late_p50=round(float(np.median(late)), 2) if late else None,
    )


def main(argv: list[str]) -> None:
    src, secs = argv[0], float(argv[1])
    people = detect_people(src, secs)
    for label, path in (a.split("=", 1) for a in argv[2:]):
        with open(path) as f:
            result = metrics(json.load(f), people)
        print("METRIC", src.split("/")[-1], label, json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
