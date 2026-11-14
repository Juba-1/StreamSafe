"""Tester session report: one short file per stream, in Arabic and English, that a tester sends back to us.

Built only from the run summary (counts and timings). It holds no pictures, no stream URL or key and no file
paths, so a streamer can share it without leaking anything. Written next to the JSON so we can also read it by code.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from streamsafe import __version__
from streamsafe.branding import PRODUCT_NAME

# full-picture covers that are not the normal per-character cover
WARN_FAILCLOSED_S = 5.0  # more than this many seconds of fail-closed cover in a stream is worth a look
WARN_P95_MS = 180.0  # the realtime decision budget (headroom.limit_ms)


def default_dir() -> Path:
    from streamsafe.app import paths

    return paths.reports_dir()


def _secs(report: dict, key: str, fps: float) -> float:
    return round(report.get("summary", {}).get(key, 0) / fps, 1) if fps else 0.0


def build(report: dict[str, Any], fps: float) -> dict[str, Any]:
    s = report.get("summary", {})
    risky = report.get("risky", {})
    total = report.get("ms_total", {})
    problems = []
    if s.get("analyzer_errors"):
        problems.append("analyzer_errors")
    if s.get("worker_restarts"):
        problems.append("worker_restarts")
    if _secs(report, "failclosed_frames", fps) > WARN_FAILCLOSED_S:
        problems.append("long_failclosed")
    if total.get("p95", 0) > WARN_P95_MS:
        problems.append("slow_decisions")
    if s.get("dropped_frames"):
        problems.append("dropped_frames")
    if s.get("frames_out", -1) == 0:
        problems.append("never_started")
    return dict(
        app_version=__version__,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        os=f"{platform.system()} {platform.release()}",
        device=s.get("gpu") or s.get("device") or "",
        mode=s.get("mode", ""),
        minutes=round(s.get("seconds", 0) / 60, 1),
        decisions_per_s=s.get("decisions_per_s", 0),
        decision_ms_p95=total.get("p95"),
        risky_decisions=round(risky.get("mean", 0) * risky.get("n", 0)) if risky else 0,
        full_cover_s=_secs(report, "scene_frames", fps),
        failclosed_s=_secs(report, "failclosed_frames", fps),
        held_s=_secs(report, "held_frames", fps),
        muted_s=_secs(report, "muted_frames", fps),
        dropped_frames=s.get("dropped_frames", 0),
        worker_restarts=s.get("worker_restarts", 0),
        analyzer_errors=s.get("analyzer_errors", 0),
        verdict="ok" if not problems else "check",
        problems=problems,
    )


AR_PROBLEMS = {
    "analyzer_errors": "حصلت أخطاء في التحليل (البث كان مغطى وقتها)",
    "worker_restarts": "محرك التحليل أعاد التشغيل أثناء البث",
    "long_failclosed": "الشاشة تغطت بالكامل لفترة طويلة بسبب بطء أو خطأ",
    "slow_decisions": "الجهاز أبطأ من المطلوب في بعض اللحظات",
    "dropped_frames": "فيه فريمات ما وصلت للبث",
    "never_started": "البث ما بدأ: البرنامج وقف قبل ما توصله صورة (تأكد من OBS ومن تبويب الفحص)",
}


def text(r: dict[str, Any]) -> str:
    ar_verdict = "كل شي طبيعي" if r["verdict"] == "ok" else "يحتاج مراجعة"
    lines = [
        f"تقرير جلسة {PRODUCT_NAME} / {PRODUCT_NAME} session report",
        f"النسخة / version: {r['app_version']}   التاريخ / date: {r['created']}",
        f"الجهاز / device: {r['device']}   النظام / OS: {r['os']}",
        f"مدة البث / duration: {r['minutes']} min",
        f"قرارات بالثانية / decisions per second: {r['decisions_per_s']}   أبطأ 5% / p95: {r['decision_ms_p95']} ms",
        f"لقطات فيها شخصيات تحتاج تغطية / risky decisions: {r['risky_decisions']}",
        f"تغطية كاملة للمشهد / full-scene cover: {r['full_cover_s']} s",
        f"تغطية احتياطية (خطأ أو بطء) / fail-closed cover: {r['failclosed_s']} s",
        f"صوت مكتوم / muted audio: {r['muted_s']} s",
        f"النتيجة / verdict: {ar_verdict} ({r['verdict']})",
    ]
    for p in r["problems"]:
        lines.append(f"  - {AR_PROBLEMS.get(p, p)} ({p})")
    lines.append(f"أرسل هذا الملف لفريق {PRODUCT_NAME}. ما فيه صور ولا مفتاح البث.")
    lines.append(f"Send this file to the {PRODUCT_NAME} team; it has no images or stream key.")
    return "\n".join(lines) + "\n"


def write(report: dict[str, Any], fps: float, folder: str | os.PathLike | None = None) -> Path:
    out = Path(folder) if folder else default_dir()
    out.mkdir(parents=True, exist_ok=True)
    r = build(report, fps)
    stem = "session_" + time.strftime("%Y%m%d_%H%M%S")
    (out / f"{stem}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    path = out / f"{stem}.txt"
    path.write_text(text(r), encoding="utf-8")
    return path
