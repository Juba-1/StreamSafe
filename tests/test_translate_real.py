"""The real models (OCR, opus-mt-en-ar, fonts) end to end. Needs the translation pack: set STREAMSAFE_TRANSLATE_PACK to a
folder made by tools/build_translate_pack.py (CI builds one in the "translate" workflow); skipped where it is not there.
Writes numbers and a picture to STREAMSAFE_TRANSLATE_REPORT (CI uploads them)."""

from __future__ import annotations

import difflib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from streamsafe.config import Config
from streamsafe.translate import pack
from streamsafe.translate.arabic import blend
from streamsafe.translate.pipeline import load_pipeline
from streamsafe.translate.service import default_factory, region_rect
from tests import translate_stubs as ts
from tests.stubs import stub_analyzer
from tools import translate_demo as demo

PACK = os.environ.get("STREAMSAFE_TRANSLATE_PACK", "")
REPORT = Path(os.environ.get("STREAMSAFE_TRANSLATE_REPORT", "") or ".")
pytestmark = pytest.mark.skipif(
    not PACK or pack.problem(PACK, pins={}, full=True) != "", reason="the translation pack is not available (STREAMSAFE_TRANSLATE_PACK)"
)

SENTENCES = {
    "Follow me to the private room.": ("الغرفة الخاصة",),
    "Press X to end the dance": ("الرقص", "X"),
    "Get in the car, we are leaving!": ("السيارة",),
    "Where have you been all night, Lucia?": ("الليل", "لوسيا"),
}


def save_report(name: str, data: dict) -> None:
    if os.environ.get("STREAMSAFE_TRANSLATE_REPORT"):
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


@pytest.fixture(scope="module")
def pipe():
    p = load_pipeline(Path(PACK))
    p.warmup()
    return p


def test_real_ocr_reads_game_style_subtitles(pipe):
    font = demo.find_font()
    size = (1920, 1080)
    rect = region_rect("subtitles", size)
    for text in SENTENCES:
        frame = demo.subtitle(demo.scene(size), text, font)
        crop = frame[rect[1] : rect[3], rect[0] : rect[2]]
        import cv2 as _cv

        padded = _cv.copyMakeBorder(
            _cv.resize(crop, (1280, round(crop.shape[0] * 1280 / crop.shape[1]))), 24, 24, 24, 24, _cv.BORDER_CONSTANT
        )
        lines = pipe.reader.read(padded)
        got = " ".join(ln.text for ln in lines)
        assert difflib.SequenceMatcher(None, got.lower(), text.lower()).ratio() > 0.9, (text, got)


def test_real_translation_is_arabic_and_sensible(pipe):
    out = pipe.translator.translate(list(SENTENCES))
    for (en, must), ar in zip(SENTENCES.items(), out, strict=True):
        assert any("؀" <= c <= "ۿ" for c in ar), (en, ar)
        assert ">>" not in ar and "<" not in ar, (en, ar)
        assert all(m in ar for m in must), (en, ar)
    again = pipe.translator.translate(list(SENTENCES))
    assert again == out and pipe.translator.hits >= len(SENTENCES), "repeated lines come from the cache"
    save_report("translations.json", dict(zip(SENTENCES, out, strict=True)))


def test_real_pipeline_covers_the_english_and_writes_the_sample_picture(pipe, tmp_path):
    out = REPORT / "translate_sample.png" if os.environ.get("STREAMSAFE_TRANSLATE_REPORT") else tmp_path / "sample.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = demo.main(PACK, str(out))
    assert out.stat().st_size > 10_000 and all(r["patches"] == 1 for r in report["lines"])
    font = demo.find_font()
    size = (1920, 1080)
    rect = region_rect("subtitles", size)
    frame = demo.subtitle(demo.scene(size), "Follow me to the private room.", font)
    got = pipe({"subtitles": (frame[rect[1] : rect[3], rect[0] : rect[2]].copy(), rect)}, size)["subtitles"]
    shown = frame.copy()
    for p in got:
        blend(shown, p)
    ys, xs = np.nonzero((frame > 235).all(axis=2)[int(1080 * 0.85) : int(1080 * 0.95)])  # the white letters of the English
    ys = ys + int(1080 * 0.85)
    assert len(xs) > 500
    inside = np.zeros(len(xs), bool)
    for p in got:
        h, w = p.alpha.shape
        inside |= (xs >= p.x) & (xs < p.x + w) & (ys >= p.y) & (ys < p.y + h)
    assert inside.all(), "every letter of the English lies under the opaque Arabic box"
    assert (got[0].alpha == 255).all()


def test_real_realtime_run_and_its_cost(tmp_path):
    """The engine on its clock with the real models: Arabic on the frames of the English, and what it costs.
    Cost = (CPU seconds of this process and its children with translation) - (without), per second of video."""
    from streamsafe.runtime import runner

    shows = {(i, i + 75): t for i, t in zip(range(60, 700, 90), list(SENTENCES) * 2, strict=False)}
    n = 780
    size = (1280, 720)

    def make(f: int) -> np.ndarray:
        frame = ts.subtitle_frames(1, {}, size=size, person=False)[0]
        cv2.ellipse(frame, (400 + 3 * f % 800, 300), (50, 160), 0, 0, 360, (40, 40, 200), -1)
        for (a, b), text in shows.items():
            if a <= f < b:
                ts.draw_subtitle(frame, text, 1.0)
        return frame

    def run(on: bool):
        cfg = Config.from_env(env={}, mode="realtime", delay=2.0, translate_ar=on, translate_dir=PACK)
        sink = ts.RecordingSink()
        held = {}
        real = runner.start_translation

        def spy(c, size, factory=None):
            held["svc"] = real(c, size, factory)
            return held["svc"]

        source = ts.LazySource(n, make, size)
        if on:
            inner = source.frames

            def paced():
                t0 = time.monotonic()
                while not (held["svc"].ready or held["svc"].dead) and time.monotonic() - t0 < 120:
                    time.sleep(0.05)
                yield from inner()

            source.frames = paced  # type: ignore[method-assign]
        runner.start_translation = spy
        c0 = os.times()
        try:
            report = runner.run(cfg, source, sink, factory=stub_analyzer, translate_factory=default_factory if on else None)
        finally:
            runner.start_translation = real
        c1 = os.times()
        cpu = (
            (c1.user - c0.user)
            + (c1.system - c0.system)
            + (c1.children_user - c0.children_user)
            + (c1.children_system - c0.children_system)
        )
        return report, sink, cpu

    _, off, cpu_off = run(False)
    on_report, on, cpu_on = run(True)
    t = on_report["summary"]["translate"]
    import hashlib

    ref = [hashlib.sha1(make(v).tobytes()).hexdigest() for v in range(n)]  # the untouched source frames
    english = {v for (a, b) in shows for v in range(a, b)}
    off_diff = sorted(v for v in range(n) if off.hashes[v] != ref[v])
    changed = {v for v in range(n) if on.hashes[v] != ref[v]}
    assert not (changed - english), (
        f"Arabic only where there is English; extra {sorted(changed - english)[:10]}, "
        f"run without translation differs from the source on {off_diff[:10]} ({len(off_diff)} frames)"
    )
    assert len(changed & english) >= 0.9 * len(english), f"Arabic on {len(changed & english)} of {len(english)} English frames"
    assert on_report["summary"]["dropped_frames"] == 0 and on_report["late_frames"]["max"] == 0, "the stream stays on time"
    seconds = n / 30
    cost = dict(
        video_seconds=seconds,
        added_cpu_ms_per_s=round(1000 * (cpu_on - cpu_off) / seconds, 1),
        reads=t["reads_submitted"],
        reads_per_s=round(t["reads_submitted"] / seconds, 2),
        english_frames=len(english),
        frames_with_arabic=len(changed & english),
        frame_loop_ms=dict(observe=t["observe_ms"], overlay=t["overlay_ms"]),
        worker_ms=t["worker_ms"],
        python=sys.version.split()[0],
        cpus=os.cpu_count(),
    )
    print("TRANSLATE_COST " + json.dumps(cost))
    save_report("cost.json", cost)
    assert cost["added_cpu_ms_per_s"] < 700, "well under one core"
