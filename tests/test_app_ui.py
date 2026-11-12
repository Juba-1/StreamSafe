"""The window itself (needs a screen: runs on the Windows runner, and on Linux under xvfb; skipped otherwise)."""

import os
import sys

import pytest

tk = pytest.importorskip("tkinter")

from streamsafe.app import controller, engine, heads, i18n, modelpack, updates  # noqa: E402
from tests.appfix import FakeProtector  # noqa: E402


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
            t = w.cget("text")
            if t:
                out.append(str(t))
        except tk.TclError:
            pass
        out += texts(w)
    return out


def test_arabic_is_the_default_and_laid_out_right_to_left(app):
    assert app.lang == "ar" and app.badge.cget("text") == "متوقف"
    assert app.hint.cget("anchor") == "e" and app.hint.cget("justify") == "right"
    assert [app.nb.tab(i, "text") for i in range(3)] == ["الفحص", "الإعدادات", "الرئيسية"]  # first tab at the right
    assert app.nb.tab(app.nb.select(), "text") == "الرئيسية"
    assert i18n.t("ar", "start") in texts(app.home)
    assert app.banner.cget("text").startswith("نسخة تجريبية")


def test_english_toggle_keeps_what_was_typed(app):
    app._vars["rtmp_url"].set("rtmps://live.example/app")
    app._vars["key"].set("secret_key_value")
    app._vars["srt_port"].set("9123")
    app.toggle_lang()
    app.update()
    assert app.lang == "en" and app.badge.cget("text") == "Stopped" and app.hint.cget("anchor") == "w"
    assert (
        app.ctrl.settings.lang == "en" and app.ctrl.settings.rtmp_url == "rtmps://live.example/app" and app.ctrl.settings.srt_port == 9123
    )
    assert app._vars["key"].get() == "secret_key_value"
    app.toggle_lang()
    assert app.lang == "ar" and app.badge.cget("text") == "متوقف"


def test_key_is_masked_and_never_saved_to_settings(app):
    entries = [w for w in app.sets.winfo_children() for w in w.winfo_children() if isinstance(w, tk.Entry)]
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
    assert not app.ctrl.engine.running and "ما يمديك تبدأ" in app.message.cget("text")


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
    assert "جهازك يشتغل" in app.verdict.cget("text")


def test_expired_beta_disables_start_and_says_why(app, monkeypatch):
    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: True)
    monkeypatch.setattr(updates, "days_left", lambda today=None, expires=None: 0)
    app.show_status(force=True)
    app.update()
    assert "disabled" in app.start_btn.state()
    assert app.badge.cget("text") == "غير جاهز" and "انتهت" in app.banner.cget("text")
    app.on_start()
    assert not app.ctrl.engine.running


def test_status_words_follow_the_engine(app, monkeypatch):
    for status, word in ((engine.PROTECTED, "محمي"), (engine.COVERED, "مغطى بالكامل")):
        monkeypatch.setattr(app.ctrl, "status", lambda s=status: s)
        app.show_status(force=True)
        assert app.badge.cget("text") == word
    app.toggle_lang()
    monkeypatch.setattr(app.ctrl, "status", lambda: engine.PROTECTED)
    app.show_status(force=True)
    assert app.badge.cget("text") == "Protected"
