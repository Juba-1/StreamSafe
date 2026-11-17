"""The window itself (needs a screen: runs on the Windows runner, and on Linux under xvfb; skipped otherwise)."""

import os
import sys

import pytest

tk = pytest.importorskip("tkinter")

from streamsafe.app import controller, engine, heads, i18n, ingest, modelpack, updates  # noqa: E402
from tests.appfix import FakeProtector  # noqa: E402


def clean(s):
    return str(s).replace("\u202b", "").replace("\u202c", "")


def have_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY"))


pytestmark = pytest.mark.skipif(not have_display(), reason="no display")


@pytest.fixture
def app(tmp_path, monkeypatch):
    from streamsafe.app import ui

    monkeypatch.setattr(controller.Controller, "run_check", lambda self, cancel=None: self.check_result)
    monkeypatch.setattr(controller.Controller, "check_updates", lambda self: None)
    monkeypatch.setattr(modelpack, "problem", lambda models, quick=True: "the models are not installed yet")
    monkeypatch.setattr(heads, "why_not", lambda env=None, app_folder=None: "")
    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: False)
    monkeypatch.setattr(updates, "days_left", lambda today=None, expires=None: 12)
    ctrl = controller.Controller({"STREAMSAFE_DATA_DIR": str(tmp_path / "data")}, FakeProtector())
    try:
        a = ui.App(ctrl)
    except tk.TclError as exc:
        pytest.skip(f"no usable display: {exc}")
    a.update()
    yield a
    a.destroy()


def texts(widget):
    out = []
    for w in widget.winfo_children():
        try:
            t = clean(w.cget("text"))
            if t:
                out.append(clean(str(t)))
        except tk.TclError:
            pass
        out += texts(w)
    return out


def test_arabic_is_the_default_and_laid_out_right_to_left(app):
    assert app.lang == "ar" and clean(app.badge.cget("text")) == "متوقف"
    assert app.hint.cget("anchor") == "e" and app.hint.cget("justify") == "right"
    assert [clean(app.nb.tab(i, "text")) for i in range(3)] == ["الفحص", "الإعدادات", "الرئيسية"]  # first tab at the right
    assert clean(app.nb.tab(app.nb.select(), "text")) == "الرئيسية"
    assert i18n.t("ar", "start") in texts(app.home)
    assert clean(app.banner.cget("text")).startswith("نسخة تجريبية")


def test_english_toggle_keeps_what_was_typed(app):
    app._vars["rtmp_url"].set("rtmps://live.example/app")
    app._vars["key"].set("secret_key_value")
    app._vars["srt_port"].set("9123")
    app.toggle_lang()
    app.update()
    assert app.lang == "en" and clean(app.badge.cget("text")) == "Stopped" and app.hint.cget("anchor") == "w"
    assert (
        app.ctrl.settings.lang == "en" and app.ctrl.settings.rtmp_url == "rtmps://live.example/app" and app.ctrl.settings.srt_port == 9123
    )
    assert app._vars["key"].get() == "secret_key_value"
    app.toggle_lang()
    assert app.lang == "ar" and clean(app.badge.cget("text")) == "متوقف"


def test_key_is_masked_and_never_saved_to_settings(app):
    def all_widgets(w):
        for c in w.winfo_children():
            yield c
            yield from all_widgets(c)

    entries = [w for w in all_widgets(app.sets) if isinstance(w, tk.Entry)]
    masked = [e for e in entries if str(e.cget("show")) == "•"]
    assert len(masked) == 1
    app._vars["key"].set("never_in_settings_123")
    app.collect()
    app.ctrl.save_settings()
    from streamsafe.app import paths

    assert "never_in_settings_123" not in paths.settings_file(app.ctrl.env).read_text(encoding="utf-8")


def test_models_missing_shows_setup_and_start_is_refused_with_a_reason(app):
    app.ctrl.check_result = controller.CheckResult("ok", [{"name": "gpu", "status": "ok", "detail": "AMD Radeon"}])
    app.show_check()
    app.update()
    assert app.setup_box.winfo_ismapped()
    assert any("النماذج غير مثبتة" in t for t in texts(app.setup_box))
    app.on_start()
    assert not app.ctrl.engine.running and "ما يمديك تبدأ" in clean(app.message.cget("text"))


def test_check_list_shows_each_item_with_a_plain_word(app):
    app.ctrl.check_result = controller.CheckResult(
        "warn",
        [
            {"name": "gpu", "status": "ok", "detail": "NVIDIA RTX"},
            {"name": "ram", "status": "warn", "detail": "6 GB", "fix_ar": "الذاكرة قليلة", "fix_en": "Low memory"},
            {"name": "ffmpeg", "status": "fail", "detail": "not found", "fix_ar": "ملف FFmpeg ناقص", "fix_en": "missing"},
        ],
    )
    app.show_check()
    t = texts(app.check_list)
    assert any("كرت الشاشة" in x and "NVIDIA RTX" in x for x in t) and "الذاكرة قليلة" in t and "ملف FFmpeg ناقص" in t
    assert "جهازك يشتغل" in clean(app.verdict.cget("text"))


def test_expired_beta_disables_start_and_says_why(app, monkeypatch):
    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: True)
    monkeypatch.setattr(updates, "days_left", lambda today=None, expires=None: 0)
    app.show_status(force=True)
    app.update()
    assert "disabled" in app.start_btn.state()
    assert clean(app.badge.cget("text")) == "غير جاهز" and "انتهت" in clean(app.banner.cget("text"))
    app.on_start()
    assert not app.ctrl.engine.running


def test_status_words_follow_the_engine(app, monkeypatch):
    for status, word in ((engine.PROTECTED, "محمي"), (engine.COVERED, "مغطى بالكامل")):
        monkeypatch.setattr(app.ctrl, "status", lambda s=status: s)
        app.show_status(force=True)
        assert clean(app.badge.cget("text")) == word
    app.toggle_lang()
    monkeypatch.setattr(app.ctrl, "status", lambda: engine.PROTECTED)
    app.show_status(force=True)
    assert clean(app.badge.cget("text")) == "Protected"


def test_settings_offer_every_way_in_and_out(app):
    t = " ".join(texts(app.sets))
    for key in ("in_srt", "in_camera", "in_file", "out_rtmp", "out_record", "out_window", "test_link", "obs_guide"):
        assert clean(i18n.t("ar", key)) in t, key


def test_pasting_a_kick_address_with_the_key_fills_both_boxes(app):
    app._vars["rtmp_url"].set("rtmps://fa723fc1b171.global-contribute.live-video.net/app/sk_us-west-2_abcDEF123")
    app._vars["key"].set("")
    app.tidy_target()
    assert app._vars["rtmp_url"].get() == "rtmps://fa723fc1b171.global-contribute.live-video.net:443/app"
    assert app._vars["key"].get() == "sk_us-west-2_abcDEF123"
    assert app.platform_value() == "kick"


def test_platform_choice_fills_its_server(app):
    app._vars["rtmp_url"].set("")
    app._vars["platform"].set(app.tr("plat_youtube"))
    app.on_platform()
    assert app._vars["rtmp_url"].get() == "rtmps://a.rtmps.youtube.com:443/live2"
    app._vars["platform"].set(app.tr("plat_twitch"))
    app.on_platform()  # a preset is replaced by the next preset
    assert app._vars["rtmp_url"].get() == ingest.PRESETS["twitch"]
    app._vars["platform"].set(app.tr("plat_tiktok"))
    app.on_platform()
    assert app._vars["rtmp_url"].get() == "" and "LIVE Producer" in clean(app.platform_hint.cget("text"))


def test_link_panel_says_what_obs_and_the_platform_do(app, monkeypatch):
    monkeypatch.setattr(type(app.ctrl.engine), "running", property(lambda self: True))
    app.ctrl.settings.input, app.ctrl.settings.output = "srt", "rtmp"
    app.ctrl.link.take({"event": "waiting_obs", "port": 9000})
    app.show_link()
    assert "srt://127.0.0.1:9000" in clean(app.lk_in.cget("text"))
    app.ctrl.link.take({"event": "obs_connected"})
    app.ctrl.link.take({"event": "platform_lost", "reason": "broken", "retry_s": 4})
    app.show_link()
    assert "OBS" in clean(app.lk_in.cget("text")) and "4" in clean(app.lk_out.cget("text"))


def test_quality_and_delay_fields(app):
    app._vars["resolution"].set("1440p")
    app._vars["fps"].set("30")
    app._vars["delay"].set(app.tr("delay_auto"))
    app.collect()
    st = app.ctrl.settings
    assert (st.resolution, st.fps, st.delay) == ("1440p", 30, 0.0)
    app._vars["delay"].set("3")
    app.collect()
    assert app.ctrl.settings.delay == 3.0


def test_obs_sending_more_than_we_keep_is_said(app):
    app.ctrl.settings.resolution, app.ctrl.settings.fps = "1080p", 30
    app.ctrl.link.sent = (2560, 1440, 60.0, 30000)
    assert "1440p60" in clean(app.format_warning())
    app.ctrl.link.sent = (1920, 1080, 30.0, 4000)
    assert "4000" in clean(app.format_warning())
    app.ctrl.link.sent = (1920, 1080, 30.0, 25000)
    assert app.format_warning() == ""


def test_logo_is_ours_on_the_normal_plan(app):
    t = " ".join(texts(app.sets))
    assert clean(i18n.t("ar", "brand_pro_only")) in t
