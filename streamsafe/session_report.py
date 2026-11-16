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
    reason = s.get("stop_reason", "")
    if reason in FAILED_REASONS:
        problems.append(reason)
    return dict(
        app_version=__version__,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        os=f"{platform.system()} {platform.release()}",
        device=_device(s),
        stop_reason=reason,
        stop_detail=s.get("stop_detail", ""),
        input_connections=s.get("input_connections"),
        platform_reconnects=s.get("platform_reconnects", 0),
        unsent_frames=s.get("unsent_frames", 0),
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


FAILED_REASONS = ("input_failed", "platform_failed", "error", "start_failed")

AR_REASONS = {
    "user": "أوقفته أنت",
    "input_ended": "انتهى الدخل (OBS وقف الإرسال أو خلص الملف)",
    "input_failed": "ما قدرنا نفتح الدخل (المنفذ مشغول أو الجهاز غير موجود)",
    "platform_failed": "المنصة ما قبلت البث (العنوان أو المفتاح غلط، أو ما فيه إنترنت)",
    "error": "خطأ داخل البرنامج",
    "start_failed": "المحرك توقف قبل ما يبدأ",
}


def _device(summary: dict) -> str:
    """The graphics card of the run; for a run that ended before the models loaded, the card the PC has (so a
    report of a failed start still says which machine it was)."""
    named = summary.get("gpu") or summary.get("device") or ""
    if named and named != "?":
        return str(named)
    try:
        from streamsafe import syscheck

        gpus = syscheck._gpus()
        return ", ".join(g.name for g in gpus) + " (not loaded)" if gpus else ""
    except Exception:
        return ""


AR_PROBLEMS = {
    "input_failed": "الدخل ما انفتح: تأكد إن المنفذ نفسه في OBS وما فيه برنامج ثاني ماسكه",
    "platform_failed": "المنصة رفضت البث: راجع عنوان البث والمفتاح",
    "error": "البرنامج توقف بسبب خطأ",
    "start_failed": "المحرك توقف قبل ما يبدأ البث",
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
    if r.get("stop_reason"):
        why = r["stop_reason"]
        lines.append(f"سبب التوقف / stopped because: {AR_REASONS.get(why, why)} ({why}) {r.get('stop_detail', '')}".rstrip())
        lines.append(
            f"اتصالات OBS / OBS connections: {r.get('input_connections')}   "
            f"إعادة الاتصال بالمنصة / platform reconnects: {r.get('platform_reconnects', 0)}   "
            f"فريمات ما انرسلت / unsent frames: {r.get('unsent_frames', 0)}"
        )
    for p in r["problems"]:
        lines.append(f"  - {AR_PROBLEMS.get(p, p)} ({p})")
    lines.append(f"أرسل هذا الملف لفريق {PRODUCT_NAME}. ما فيه صور ولا مفتاح البث.")
    lines.append(f"Send this file to the {PRODUCT_NAME} team; it has no images or stream key.")
    return "\n".join(lines) + "\n"


def write(report: dict[str, Any], fps: float, folder: str | os.PathLike | None = None, log: list[str] | None = None) -> Path:
    """``log``: the engine's last lines (already without the key), added under the report (``add_log``)."""
    out = Path(folder) if folder else default_dir()
    out.mkdir(parents=True, exist_ok=True)
    r = build(report, fps)
    stem = "session_" + time.strftime("%Y%m%d_%H%M%S")
    n = 1
    while (out / f"{stem}.txt").exists():  # two reports in one second (a start that fails at once, twice)
        n += 1
        stem = "session_" + time.strftime("%Y%m%d_%H%M%S") + f"_{n}"
    (out / f"{stem}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    path = out / f"{stem}.txt"
    path.write_text(text(r), encoding="utf-8")
    if log:
        add_log(path, log)
    return path


LOG_HEAD = "--- سجل المحرك (آخر الأسطر) / engine log (last lines) ---"


def add_log(path: Path, lines: list[str], keep: int = 40) -> None:
    """Put the engine's last lines under a report (once): a failed start then explains itself in one file."""
    try:
        if LOG_HEAD in path.read_text(encoding="utf-8"):
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + LOG_HEAD + "\n" + "\n".join(line[:300] for line in lines[-keep:]) + "\n")
    except OSError:
        pass


def write_failure(reason: str, detail: str, lines: list[str], folder: str | os.PathLike | None = None) -> Path:
    """The report of an engine that ended without writing one (it was refused at start, crashed or was killed)."""
    summary = {"frames_out": 0, "seconds": 0, "analyzer_errors": 0, "stop_reason": reason, "stop_detail": detail[:400]}
    return write({"summary": summary}, 30.0, folder, lines)
