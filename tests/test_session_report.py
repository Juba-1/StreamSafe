from streamsafe import session_report


def _report(**summary):
    base = dict(
        seconds=600,
        decisions_per_s=10.0,
        mode="realtime",
        gpu="onnx-trt",
        scene_frames=90,
        failclosed_frames=30,
        held_frames=0,
        muted_frames=90,
        dropped_frames=0,
        worker_restarts=0,
        analyzer_errors=0,
    )
    base.update(summary)
    return dict(summary=base, ms_total=dict(p95=88.0), risky=dict(n=100, mean=0.25))


def test_clean_session_is_ok_and_has_no_secrets(tmp_path):
    path = session_report.write(_report(), 30.0, tmp_path)
    txt = path.read_text(encoding="utf-8")
    r = session_report.build(_report(), 30.0)
    assert r["verdict"] == "ok" and r["full_cover_s"] == 3.0 and r["failclosed_s"] == 1.0 and r["risky_decisions"] == 25
    assert "rtmp" not in txt and "تقرير" in txt and path.with_suffix(".json").exists()


def test_problems_are_flagged():
    r = session_report.build(_report(failclosed_frames=300, worker_restarts=1, analyzer_errors=2), 30.0)
    assert r["verdict"] == "check" and {"long_failclosed", "worker_restarts", "analyzer_errors"} <= set(r["problems"])
