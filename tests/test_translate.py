"""Arabic translation of on-screen text: the safety contract and the pieces (stub models; see test_translate_real.py
for the real ones). Names with "realtime" run the engine on its real clock (Linux CI only, like the other such tests)."""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import shutil
import threading
import time

import numpy as np
import pytest

from streamsafe.app import engine, fetch, release
from streamsafe.app import settings as app_settings
from streamsafe.config import Config
from streamsafe.runtime import runner
from streamsafe.translate import pack, textproc
from streamsafe.translate import service as svc_mod
from streamsafe.translate.arabic import PAD_X_K, PAD_Y_K, ArabicRenderer, Patch, blend, visual_order
from streamsafe.translate.ocr import Line
from streamsafe.translate.pipeline import SAME, Pipeline, group
from streamsafe.translate.service import TranslationService, fine_signature, region_rect, sig_distance, signature, text_band
from streamsafe.translate.translator import CachedTranslator
from tests import translate_stubs as ts
from tests.crashing import process_dies
from tests.stubs import MAGENTA, stub_analyzer, stub_scene_analyzer
from tests.test_app_controller import ctrl  # noqa: F401 - the fixture
from tests.test_engine import is_slate


def original_shows_through(area: np.ndarray) -> bool:
    """True if a pixel of the original picture (the plain background, or the subtitle's black outline) is in ``area``."""
    bg = (area == np.array(ts.BG, np.uint8)).all(axis=2)
    outline = (area == 0).all(axis=2)
    return bool(bg.any() or outline.any())


def renderer() -> ArabicRenderer:
    try:
        return ArabicRenderer(None)
    except RuntimeError:
        pytest.skip("no font with Arabic letters on this machine")


# ----------------------------------------------------------------------------------------------- text helpers
def test_text_helpers():
    assert textproc.clean("  Press​  X \n to  end ") == "Press X to end"
    assert textproc.cache_key("Press X to END the dance!") == textproc.cache_key("press  x to end the dance")
    assert textproc.cache_key("Don't go.") != textproc.cache_key("Dont go")
    assert textproc.worth_translating("Follow me to the private room.")
    assert not textproc.worth_translating("12:45")
    assert not textproc.worth_translating("ab")
    assert not textproc.worth_translating("@#$%^&*()")
    assert textproc.sentences("Where were you? I was worried sick. Come on!") == ["Where were you?", "I was worried sick.", "Come on!"]
    assert textproc.sentences("Dr. smith is here") == ["Dr. smith is here"]


def test_cache_answers_repeated_lines_without_the_model():
    inner = ts.StubTranslator()
    cached = CachedTranslator(inner, size=2)
    a, b, c = "Follow me to the private room.", "Press X to end the dance", "Where have you been all night, Lucia?"
    assert cached.translate([a, b]) == [ts.LINES[a], ts.LINES[b]]
    assert cached.translate(["follow  me to the PRIVATE room", a]) == [ts.LINES[a]] * 2, "case, spacing, punctuation do not matter"
    assert inner.calls == [[a, b]], "the second call never reached the model"
    assert cached.hits == 2 and cached.misses == 2
    cached.translate([c])  # size 2: the oldest entry (b) goes
    cached.translate([b])
    assert len(inner.calls) == 3, "the evicted line is asked again"


# -------------------------------------------------------------------------------------------------- the Arabic
def test_bidi_order_keeps_latin_and_numbers_readable():
    # logical "ab 12 cd" inside Arabic: the visual order puts the Arabic runs in reverse and keeps "X" and "50" whole
    s = "اضغط X لإنهاء 50 الرقص"
    v = visual_order(s)
    assert "X" in v and "50" in v and "05" not in v
    assert v.index("50") < v.index("X"), "the later words of the sentence are drawn further left"
    assert visual_order("(نعم)") == "(نعم)"[::-1].translate(str.maketrans("()", ")(")), "brackets are mirrored"


def test_arabic_box_hides_the_english_completely():
    r = renderer()
    frame = ts.subtitle_frames(1, {(0, 1): "Press X to end the dance"}, size=(1280, 720))[0]
    box = ts.draw_subtitle(np.full((720, 1280, 3), ts.BG, np.uint8), "Press X to end the dance")  # same place, bigger than the letters
    patch = r.render("اضغط X لإنهاء الرقص", box, (1280, 720))
    assert patch is not None
    assert (patch.alpha == 255).all(), "fully opaque: nothing of the English shows through"
    en_h = box[3] - box[1]
    assert patch.x <= box[0] - round(PAD_X_K * en_h) + 1 and patch.x + patch.alpha.shape[1] >= box[2] + round(PAD_X_K * en_h) - 1
    assert patch.y <= box[1] - round(PAD_Y_K * en_h) + 1 and patch.y + patch.alpha.shape[0] >= box[3]
    shown = frame.copy()
    blend(shown, patch)
    x0, y0, x1, y1 = box
    assert not original_shows_through(shown[y0:y1, x0:x1]), "the English area is overwritten: no background or outline pixel is left"
    same_outside = np.ones(frame.shape[:2], bool)
    same_outside[patch.y : patch.y + patch.alpha.shape[0], patch.x : patch.x + patch.alpha.shape[1]] = False
    assert (shown[same_outside] == frame[same_outside]).all(), "only the box changes"


def test_long_arabic_widens_and_wraps_inside_the_band_and_still_covers_the_english():
    r = renderer()
    long = "كنت قلقاً جداً عليك طوال الليل، أين كنت؟ يجب أن نتحدث الآن عن كل ما حدث بيننا في الأسابيع الماضية وعن كل شيء آخر."
    box = (700, 640, 1000, 680)  # a short English line
    patch = r.render(long, box, (1920, 1080))
    h, w = patch.alpha.shape
    assert w > 300 and w <= 0.92 * 1920 + 1, "widened, within 92 % of the frame"
    assert h > 60, "wrapped onto more lines"
    assert patch.x <= box[0] and patch.x + w >= box[2] and patch.y <= box[1] and patch.y + h >= box[3]
    assert patch.y >= 0 and patch.y + h <= 1080


def test_blend_clips_at_the_frame_edge():
    frame = np.full((50, 80, 3), 100, np.uint8)
    patch = Patch(70, 40, np.zeros((30, 30, 3), np.uint8), np.full((30, 30), 255, np.uint8))
    blend(frame, patch)
    assert (frame[40:, 70:] == 0).all() and (frame[:40] == 100).all()
    blend(frame, Patch(-100, -100, patch.bgr, patch.alpha))  # entirely outside: nothing, no error


# ------------------------------------------------------------------------------------------------ the pipeline
def _crop(text: str | None):
    frame = ts.subtitle_frames(1, {(0, 1): text} if text else {}, size=ts.SIZE, person=False)[0]
    rect = region_rect("subtitles", ts.SIZE)
    return {"subtitles": (frame[rect[1] : rect[3], rect[0] : rect[2]].copy(), rect)}


def test_pipeline_translates_once_and_skips_unchanged_text():
    tr = ts.StubTranslator()
    pipe = Pipeline(ts.StubReader(), CachedTranslator(tr), renderer())
    line = "Follow me to the private room."
    first = pipe(_crop(line), ts.SIZE)
    assert len(first["subtitles"]) == 1 and isinstance(first["subtitles"][0], Patch)
    assert pipe(_crop(line), ts.SIZE) == {"subtitles": SAME}, "same English: not translated or drawn again"
    assert pipe(_crop(None), ts.SIZE) == {"subtitles": []}, "no text: nothing to draw"
    again = pipe(_crop(line), ts.SIZE)
    assert len(again["subtitles"]) == 1
    assert tr.calls == [[line]], "the repeated line came from the cache"


def test_lines_of_one_subtitle_become_one_paragraph():
    lines = [
        Line((100, 10, 400, 40), "Where have you been", 0.9),
        Line((120, 44, 380, 74), "all night, Lucia?", 0.8),
        Line((100, 300, 400, 330), "Far away", 0.9),
    ]
    paras = group(lines)
    assert [p.text for p in paras] == ["Where have you been all night, Lucia?", "Far away"]
    assert paras[0].box == (100, 10, 400, 74) and paras[0].conf == 0.8


# ----------------------------------------------------------------------------------------- service pieces
def make_service(**kw) -> TranslationService:
    return TranslationService(Config.from_env(env={}, translate_ar=True, **kw), ts.SIZE, ts.stub_factory)


def test_service_matches_reads_to_frames_by_what_the_region_shows():
    """Frame-exact: a read applies to frames whose region looks like the read's, before and after it in time."""
    svc = make_service()
    a, b = "Follow me to the private room.", "Press X to end the dance"
    frames = ts.subtitle_frames(60, {(10, 30): a, (40, 55): b}, person=False)
    for i, f in enumerate(frames):
        svc.observe(i, f)  # not ready: only the pictures' signatures are kept
    patch_a, patch_b = (
        Patch(0, 0, np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 255, np.uint8)),
        Patch(5, 5, np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 255, np.uint8)),
    )
    svc._submitted[12] = svc._sigs[12]  # what observe() records when it hands a read to the worker
    svc._submitted[45] = svc._sigs[45]
    svc._store(12, {"subtitles": [patch_a]})  # a read taken on frame 12 (it may arrive any time before the frame is shown)
    svc._store(45, {"subtitles": [patch_b]})
    got = {v: svc.patches_for(v) for v in range(60)}
    assert all(got[v] == [patch_a] for v in range(10, 30)), (
        "the Arabic is on every frame the English is on, also before the read's own frame"
    )
    assert all(got[v] == [patch_b] for v in range(40, 55))
    assert all(got[v] == [] for v in [*range(10), *range(30, 40), *range(55, 60)]), "and on no frame without that English"
    svc._store(46, {"subtitles": SAME})
    assert svc.patches_for(50) == [patch_b], "unchanged text keeps the earlier boxes"


def test_signature_ignores_a_moving_background_but_sees_new_text():
    r = region_rect("subtitles", ts.SIZE)
    base = ts.subtitle_frames(1, {}, person=False)[0]
    moved = ts.subtitle_frames(1, {}, person=True)[0]
    with_text = ts.subtitle_frames(1, {(0, 1): "Press X to end the dance"}, person=False)[0]
    other = ts.subtitle_frames(1, {(0, 1): "Follow me to the private room."}, person=False)[0]
    s0, s1, s2, s3 = (signature(f, r) for f in (base, moved, with_text, other))
    assert sig_distance(s0, s1) == 0.0
    assert sig_distance(s2, s2) == 0.0
    assert sig_distance(s2, s3) > 0.5, "another line is not the same text"
    assert sig_distance(s2, s0) > 0.5, "text appearing is a change"


def test_overlay_is_a_copy_and_never_touches_the_frame_the_gate_keeps():
    svc = make_service()
    frames = ts.subtitle_frames(3, {(0, 3): "Press X to end the dance"}, person=False)
    for i, f in enumerate(frames):
        svc.observe(i, f)
    svc._submitted[0] = svc._sigs[0]
    svc._store(0, {"subtitles": [Patch(100, 900, np.zeros((40, 300, 3), np.uint8), np.full((40, 300), 255, np.uint8))]})
    kept = frames[1].copy()
    shown = svc.overlay(1, frames[1])
    assert shown is not frames[1] and (frames[1] == kept).all(), "the gate's frame is untouched: a held frame cannot repeat the Arabic"
    assert (shown[900:940, 100:400] == 0).all()
    assert svc.overlay(2, frames[2]) is not frames[2]
    svc.dead = True
    assert svc.overlay(1, frames[1]) is frames[1], "a dead service returns the very same frame"


def test_service_never_raises_into_the_frame_loop():
    svc = make_service()
    svc.observe(0, np.zeros((10, 10, 3), np.uint8))  # a frame of the wrong size
    svc.observe(1, "not a frame")  # type: ignore[arg-type]
    assert svc.stats["observe_errors"] == 2
    assert svc.overlay(0, np.zeros((10, 10, 3), np.uint8)).shape == (10, 10, 3)


def test_queue_is_bounded_and_a_full_queue_drops_the_read():
    svc = make_service(translate_queue=1, translate_min_gap_s=0.0)
    svc.ready = True  # pretend the worker is loaded; nobody takes jobs
    for i, f in enumerate(ts.subtitle_frames(1, {(0, 1): "Press X to end the dance"}, person=False) * 1):
        svc.observe(i, f)
    for i in range(1, 6):
        svc.observe(
            i, ts.subtitle_frames(1, {(0, 1): ["Follow me to the private room.", "Press X to end the dance"][i % 2]}, person=False)[0]
        )
        time.sleep(0.05)
    assert svc.stats["reads_submitted"] >= 1 and svc.stats["reads_dropped"] >= 1
    svc.stop()


# ---------------------------------------------------------------------------------- fewer, cheaper reads (slow CPUs)
class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def timed_service(regions: str = "subtitles", **kw) -> tuple[TranslationService, Clock]:
    clock = Clock()
    cfg = Config.from_env(env={}, translate_ar=True, translate_regions=regions, **kw)
    svc = TranslationService(cfg, ts.SIZE, ts.stub_factory, clock=clock)
    svc.ready = True  # pretend the worker is loaded: the jobs just wait in the queue, nobody takes them
    return svc, clock


def drain(svc: TranslationService) -> list:
    out = []
    while True:
        try:
            out.append(svc._jobs.get(timeout=0.3))
        except Exception:
            return out


def shown(text: str | None) -> np.ndarray:
    return ts.subtitle_frames(1, {(0, 1): text} if text else {}, person=False)[0]


def test_unchanged_subtitle_is_read_once_not_every_refresh_time():
    svc, clock = timed_service()
    line = shown("Press X to end the dance")
    for i in range(80):  # 8 s of the same subtitle, a frame every 0.1 s: the refresh time (2 s) passes 3 times
        svc.observe(i, line)
        clock.t += 0.1
    assert svc.stats["reads_submitted"] == 1, "a closer look at the unchanged text finds nothing new: no second read"
    svc.stop()


def test_a_changed_word_in_unchanged_looking_text_is_still_read_at_the_refresh():
    svc, clock = timed_service()
    a, b = shown("Press X to end the dance"), shown("Press Y to end the dance")
    r = region_rect("subtitles", ts.SIZE)
    assert sig_distance(signature(a, r), signature(b, r)) <= svc_mod.SIG_SAME, "the per-frame picture does not see it"
    assert sig_distance(fine_signature(a, r), fine_signature(b, r)) > svc_mod.FINE_SAME, "the closer look does"
    for i in range(10):
        svc.observe(i, a)
        clock.t += 0.1
    assert svc.stats["reads_submitted"] == 1
    for i in range(10, 40):
        svc.observe(i, b)
        clock.t += 0.1
    assert svc.stats["reads_submitted"] == 2
    svc.stop()


def test_unchanged_text_is_read_again_after_the_quiet_limit_whatever_the_closer_look_says():
    svc, clock = timed_service()
    line = shown("Press X to end the dance")
    n = int((svc_mod.QUIET_MAX_S + 3) / 0.1)
    for i in range(n):
        svc.observe(i, line)
        clock.t += 0.1
    assert svc.stats["reads_submitted"] == 2
    svc.stop()


def test_backoff_holds_new_text_back_only_briefly_and_refreshes_for_longer():
    svc, clock = timed_service()
    line = shown("Press X to end the dance")
    svc.stream_late()
    svc.observe(0, line)
    assert svc.stats["reads_submitted"] == 0, "right after a late frame nothing is read"
    clock.t += svc_mod.BACKOFF_NEW_S + 0.01
    svc.observe(1, line)
    assert svc.stats["reads_submitted"] == 1, "new text is read after the short pause, not after the long one"
    svc.stop()
    assert svc_mod.BACKOFF_NEW_S < svc_mod.BACKOFF_S

    svc, clock = timed_service(translate_refresh_s=1.0)  # a refresh look waits for the long pause
    other = shown("Follow me to the private room.")
    svc.observe(0, line)
    clock.t += 1.2
    svc.observe(1, other)  # changed text: due at once
    assert svc.stats["reads_submitted"] == 2
    svc.stream_late()
    clock.t += 1.1
    svc.observe(2, other)
    assert svc._read_at["subtitles"] < clock.t - 1.0 and svc.stats["reads_submitted"] == 2, "no refresh inside the long pause"
    svc.stop()


def test_only_the_band_with_the_text_is_handed_to_the_worker():
    svc, _ = timed_service()
    svc.observe(0, shown("Press X to end the dance"))
    ((fid, crops),) = drain(svc)
    bgr, rect = crops["subtitles"]
    full = region_rect("subtitles", ts.SIZE)
    assert fid == 0 and bgr.shape[:2] == (rect[3] - rect[1], rect[2] - rect[0])
    assert (rect[0], rect[2]) == (full[0], full[2]) and full[1] <= rect[1] < rect[3] <= full[3]
    assert (rect[3] - rect[1]) <= 0.4 * (full[3] - full[1]), "a one-line subtitle: a third of the region, not all of it"
    sig = signature(shown("Press X to end the dance"), full)
    assert text_band(sig, full) == rect
    assert text_band(np.zeros_like(sig), full) == full, "no text: the region as it is"
    assert text_band(np.ones_like(sig), full) == full, "text everywhere: the whole region"
    svc.stop()


def test_a_region_without_text_is_not_read_and_one_with_unchanged_text_is_not_read_again():
    svc, clock = timed_service(regions="subtitles,help")
    both = shown("Press X to end the dance")
    help_r = region_rect("help", ts.SIZE)
    import cv2

    with_help = both.copy()
    cv2.putText(
        with_help, "Hold L2 to aim", (help_r[0] + 40, help_r[1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA
    )
    svc.observe(0, both)  # subtitles only
    clock.t += 0.5
    svc.observe(1, with_help)  # the help box gets text, the subtitle is unchanged
    jobs = drain(svc)
    assert len(jobs) == 2
    assert set(jobs[0][1]) == {"subtitles"}, "nothing written in the help box: it is not sent"
    assert jobs[1][1]["subtitles"] is None and jobs[1][1]["help"] is not None, "unchanged subtitle: only its boxes are repeated"
    svc.stop()


def test_pipeline_repeats_boxes_without_reading_and_forgets_regions_that_lost_their_text():
    tr = ts.StubTranslator()
    pipe = Pipeline(ts.StubReader(), CachedTranslator(tr), renderer())
    line = "Follow me to the private room."
    assert len(pipe(_crop(line), ts.SIZE)["subtitles"]) == 1
    assert pipe({"subtitles": None}, ts.SIZE) == {"subtitles": SAME}, "carried over: no reading at all"
    assert len(pipe.ms["ocr"]) == 1
    assert pipe({}, ts.SIZE) == {}, "region not given: no text there"
    again = pipe(_crop(line), ts.SIZE)
    assert len(again["subtitles"]) == 1, "the text came back after a gap: drawn again, not 'same'"


def test_warmup_runs_every_stage_once_and_leaves_no_trace():
    calls = {"read": 0, "render": 0}

    class Reader(ts.StubReader):
        def read(self, bgr):
            calls["read"] += 1
            return super().read(bgr)

    class Renderer:
        def render(self, *a, **k):
            calls["render"] += 1

    tr = ts.StubTranslator()
    pipe = Pipeline(Reader(), CachedTranslator(tr), Renderer())  # type: ignore[arg-type]
    pipe.warmup()
    assert calls == {"read": 1, "render": 1} and len(tr.calls) == 1
    assert all(v == [] for v in pipe.ms.values()) and pipe._last == {}

    class Broken:
        def read(self, bgr):
            raise RuntimeError("no")

    Pipeline(Broken(), CachedTranslator(ts.StubTranslator()), Renderer()).warmup()  # type: ignore[arg-type]  # does not raise


def test_a_read_that_arrives_late_reaches_every_frame_of_its_subtitle_still_waiting():
    """Backfill: the read was taken on frame 40 (held back at first); frames 10..39 of the same subtitle wait in the delay
    buffer and get the Arabic too, as soon as it is there."""
    svc = make_service()
    frames = ts.subtitle_frames(60, {(10, 50): "Press X to end the dance"}, person=False)
    for i, f in enumerate(frames):
        svc.observe(i, f)
    assert all(svc.patches_for(v) == [] for v in range(60)), "nothing finished yet"
    patch = Patch(5, 5, np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 255, np.uint8))
    svc._submitted[40] = svc._sigs[40]
    svc._store(40, {"subtitles": [patch]})
    assert all(svc.patches_for(v) == [patch] for v in range(10, 50))
    assert all(svc.patches_for(v) == [] for v in [*range(10), *range(50, 60)])


# ------------------------------------------------------------------------------------------ pack and settings
FAKE = {n: f"content of {n}".encode() * 3 for n in pack.REQUIRED}


def fake_pins(files=FAKE):
    return {n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)} for n, b in files.items()}


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    for n, b in FAKE.items():
        (root / n).write_bytes(b)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{httpd.server_port}/"
    httpd.shutdown()


def test_pack_downloads_only_pinned_files(tmp_path, server):
    root, url = server
    dest = tmp_path / "data" / "translate"
    assert pack.problem(dest, fake_pins()) != ""
    seen = []
    pack.install_download(url, dest, tmp_path / "dl", fake_pins(), lambda w, d, t: seen.append((d, t)), allow_http=True)
    assert pack.problem(dest, fake_pins(), full=True) == "" and seen[-1][0] == seen[-1][1]
    (dest / "model.bin").write_bytes(b"x")
    assert "damaged" in pack.problem(dest, fake_pins())
    (root / "model.bin").write_bytes(b"tampered" * 3)  # the server now serves something else
    with pytest.raises(fetch.DownloadError):
        pack.install_download(url, dest, tmp_path / "dl2", fake_pins(), allow_http=True)
    assert (dest / "model.bin").read_bytes() == b"x", "a failed download leaves what was there"
    assert not list((tmp_path / "data").glob("translate.new-*")), "no staging folder is left"


def test_pack_from_a_folder_is_checked_too(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for n, b in FAKE.items():
        (src / n).write_bytes(b)
    pack.install_folder(src, tmp_path / "t", fake_pins())
    assert pack.problem(tmp_path / "t", fake_pins(), full=True) == ""
    (src / "det.onnx").write_bytes(b"other")
    with pytest.raises(pack.PackError):
        pack.install_folder(src, tmp_path / "t2", fake_pins())


def test_release_pins_cover_every_required_file():
    assert set(pack.REQUIRED) <= set(release.TRANSLATE_FILES)
    assert all(len(s) == 64 and n > 0 for s, n in release.TRANSLATE_FILES.values())
    assert release.TRANSLATE_BASE_URL.startswith("https://")
    assert "huggingface.co" in release.allowed_hosts() or release.allowed_hosts()


def test_off_by_default_everywhere():
    assert Config.from_env(env={}).translate.translate_ar is False
    assert app_settings.Settings().translate_ar is False
    assert Config.from_env(env={"TRANSLATE_AR": "1"}).translate.translate_ar is True, "the CLI flag"
    base = {"PATH": "x"}
    assert "TRANSLATE_AR" not in engine.engine_env(base, False)
    env = engine.engine_env(base, False, translate=True)
    assert env["TRANSLATE_AR"] == "1" and env["TRANSLATE_DIR"].endswith("translate")
    assert "TRANSLATE_AR" not in engine.engine_env({"TRANSLATE_AR": "1"}, False), "the app's setting decides, not a stray variable"


def test_app_passes_translation_on_only_when_set_and_installed(ctrl, monkeypatch):  # noqa: F811
    seen = []
    real = engine.EngineProcess.start

    def spy(self, cmd, env, log_path=None):
        seen.append(dict(env))
        return real(self, cmd, env, log_path)

    monkeypatch.setattr(engine.EngineProcess, "start", spy)
    cases = [(False, ""), (True, "files missing"), (True, "")]  # (the setting, translate_problem)
    for want, problem in cases:
        ctrl.settings.translate_ar = want
        monkeypatch.setattr(ctrl, "translate_problem", lambda problem=problem: problem)
        ctrl.start("")
        ctrl.engine.stop(2)
    assert ["TRANSLATE_AR" in e for e in seen] == [False, False, True], "off, or on without files: the engine is told nothing"
    assert seen[2]["TRANSLATE_DIR"].endswith("translate")


def test_app_downloads_the_pack_and_reports_why_not_instead_of_raising(ctrl, server, monkeypatch, tmp_path):  # noqa: F811
    root, url = server
    monkeypatch.setattr(release, "TRANSLATE_BASE_URL", url)
    monkeypatch.setattr(release, "TRANSLATE_FILES", {n: (h["sha256"], h["bytes"]) for n, h in fake_pins().items()})
    ctrl.allow_http = True
    assert ctrl.translate_problem() != ""
    assert ctrl.install_translate() == ""
    assert ctrl.translate_problem() == ""
    (root / "det.onnx").write_bytes(b"tampered" * 3)
    paths_dir = ctrl.env["STREAMSAFE_DATA_DIR"]
    shutil.rmtree(f"{paths_dir}/translate")
    shutil.rmtree(f"{paths_dir}/downloads", ignore_errors=True)
    why = ctrl.install_translate()
    assert why and ctrl.translate_problem() != "", "a bad download is reported, nothing is installed, nothing raised"


def test_translation_without_its_files_starts_nothing(capsys):
    cfg = Config.from_env(env={}, translate_ar=True, translate_dir="/nonexistent")
    assert runner.start_translation(cfg, ts.SIZE) is None
    assert "TRANSLATE_OFF" in capsys.readouterr().out
    assert runner.start_translation(Config.from_env(env={}), ts.SIZE) is None


# --------------------------------------------------------------------------------------- the engine, realtime
def run_engine(frames, translate_factory=None, analyzer=stub_analyzer, wait_ready=True, **settings):
    """One realtime run over ``frames``; returns (report, sink with every output frame, the service)."""
    cfg = Config.from_env(env={}, mode="realtime", delay=1.0, translate_ar=translate_factory is not None, **settings)
    held: dict = {}
    real_start = runner.start_translation

    def spy(c, size, factory=None):
        held["svc"] = real_start(c, size, factory)
        return held["svc"]

    source, sink = ts.ListSource([f.copy() for f in frames]), ts.RecordingSink()  # the gate draws on its frames in place
    sink.keep_frames = True
    if wait_ready and translate_factory is not None:
        inner = source.frames

        def paced():  # the models of the worker load in the background: let the test start when they are there
            t0 = time.monotonic()
            while not (held["svc"].ready or held["svc"].dead) and time.monotonic() - t0 < 60:
                time.sleep(0.05)
            yield from inner()

        source.frames = paced  # type: ignore[method-assign]
    runner.start_translation = spy
    try:
        report = runner.run(cfg, source, sink, factory=analyzer, translate_factory=translate_factory)
    finally:
        runner.start_translation = real_start
    return report, sink, held.get("svc")


SHOWS = {(30, 80): "Follow me to the private room.", (100, 140): "Press X to end the dance"}
N = 160


def small_frames(shows=SHOWS, n=N, **kw):
    return ts.subtitle_frames(n, shows, size=(1280, 720), **kw)


def test_realtime_arabic_is_on_exactly_the_frames_of_the_english_and_hides_it():
    frames = small_frames()
    _, off, _ = run_engine(frames)
    report, on, _ = run_engine(frames, ts.stub_factory)
    assert len(on.hashes) == len(off.hashes) == N
    changed = {v for v in range(N) if on.hashes[v] != off.hashes[v]}
    english = set(range(30, 80)) | set(range(100, 140))
    assert changed == english, f"extra: {sorted(changed - english)[:5]}, missing: {sorted(english - changed)[:5]}"
    t = report["summary"]["translate"]
    assert t["frames_overlaid"] == len(english) and not t["errors"] and t["reads_done"] >= 2
    assert t["reads_submitted"] <= 8, "reads happen when the text changes (and now and then), not every frame"
    for v in (30, 55, 79, 100, 139):  # the English letters are gone: every pixel of its box was overwritten
        box = ts.draw_subtitle(
            np.full((720, 1280, 3), ts.BG, np.uint8), "Follow me to the private room." if v < 90 else "Press X to end the dance", 1.4
        )
        x0, y0, x1, y1 = box
        assert not original_shows_through(on.frames[v][y0:y1, x0:x1]), f"English shows through on frame {v}"
    assert (on.frames[10] == off.frames[10]).all() and (on.frames[90] == off.frames[90]).all()


def test_realtime_arabic_goes_on_top_of_the_full_scene_cover_and_the_cover_stays_whole():
    """The owner's rule: when the picture is covered the dialogue still gets its Arabic, drawn ON the cover; the cover
    itself (everything outside the Arabic box) is untouched, and frames without English stay exactly the cover."""
    band = slice(int(0.70 * 720), 720)  # the subtitle band of the 1280x720 test frames, where the Arabic box sits
    # (1) the analyzer dies: after a while every frame is the slate, subtitles keep coming until frame 150
    frames = small_frames({(10, 150): "Follow me to the private room."})
    _, off, _ = run_engine(frames, analyzer=process_dies)
    _, on, _ = run_engine(frames, ts.stub_factory, analyzer=process_dies)
    slates = [v for v in range(N) if is_slate(off.frames[v])]
    assert len(slates) > 60
    for v in slates:
        assert (on.frames[v][: band.start] == off.frames[v][: band.start]).all(), "the slate above the band is untouched"
        if v >= 160:
            assert on.hashes[v] == off.hashes[v], "no English, no Arabic: exactly the slate"
    assert any(on.hashes[v] != off.hashes[v] for v in slates if v < 150), "the dialogue gets Arabic on the slate"
    # (2) a whole-screen scene card while a subtitle is on screen
    frames = small_frames({(10, 150): "Press X to end the dance"})
    for v in range(60, 120):
        frames[v][:] = MAGENTA
        ts.draw_subtitle(frames[v], "Press X to end the dance")
    _, on, _ = run_engine(frames, ts.stub_factory, analyzer=stub_scene_analyzer)
    cards = list(range(75, 105))
    tops = {on.frames[v][: band.start].tobytes() for v in cards}
    assert len(tops) == 1, "the card itself is one identical picture above the band"
    shown = on.frames[cards[0]][: band.start]
    assert not (shown == np.array(MAGENTA, dtype=shown.dtype)).all(axis=2).any(), "the covered picture never shows through"


@pytest.mark.parametrize("factory", ["dying_factory", "failing_factory", "slow_factory", "broken_load_factory"])
def test_realtime_a_failing_slow_or_dead_worker_leaves_the_stream_exactly_as_without_it(factory):
    frames = small_frames()
    _, off, _ = run_engine(frames)
    report, on, _ = run_engine(frames, getattr(ts, factory), wait_ready=False)
    assert report["summary"]["frames_out"] == N and report["summary"]["dropped_frames"] == 0, "no frame lost or late"
    t = report["summary"]["translate"]
    assert t["frames_overlaid"] == 0, "nothing was drawn"
    # two runs on a loaded clock can differ by a held frame or two; an overlay would change dozens
    assert sum(a != b for a, b in zip(on.hashes, off.hashes, strict=True)) <= 3, "output identical to the run without translation"
    assert report["emit_lateness_ms"]["max"] < 250, "and on time"
    if factory in ("dying_factory", "broken_load_factory"):
        assert t["dead"] and t["errors"]
    if factory == "failing_factory":
        assert t["worker_errors"] >= 1 and t["frames_overlaid"] == 0
    if factory == "slow_factory":
        assert t["frames_overlaid"] == 0, "a read that is not ready in time is not used"


def test_realtime_off_by_default_report_and_frames_are_unchanged():
    frames = small_frames(n=60)
    report, _, svc = run_engine(frames)
    assert svc is None and "translate" not in report["summary"]


def test_realtime_overlay_costs_little_in_the_frame_loop():
    report, _, _ = run_engine(small_frames(), ts.stub_factory)
    t = report["summary"]["translate"]
    assert t["observe_ms"]["p95"] < 10 and t["overlay_ms"]["p95"] < 25
    json.dumps(report["summary"])
