"""Entry commands (expiry / heads refusal), all window texts, and the Windows build tools."""

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from streamsafe.app import entry, heads, i18n, updates
from tests.appfix import make_heads

ROOT = Path(__file__).resolve().parents[1]


# --- entry ------------------------------------------------------------------------------------------------------
@pytest.fixture
def product_env(tmp_path, monkeypatch):
    env = {"STREAMSAFE_PRODUCT": "1", "STREAMSAFE_DATA_DIR": str(tmp_path / "data"), "PATH": os.environ.get("PATH", "")}
    monkeypatch.setattr(os, "environ", env)
    return env


def test_run_is_refused_after_the_beta_ended(product_env, monkeypatch, capsys):
    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: True)
    assert entry.main(["run", "--input", "x", "--output", "window"]) == entry.EXPIRED_EXIT == 4
    out = capsys.readouterr().out
    assert "BETA_EXPIRED" in out and "Nothing was streamed" in out


def test_run_is_refused_without_usable_heads(product_env, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: False)
    app = tmp_path / "app"
    monkeypatch.setattr(entry.paths, "app_dir", lambda: app)
    assert entry.main(["run", "--input", "x", "--output", "window"]) == entry.NO_HEADS_EXIT
    assert "HEADS_MISSING" in capsys.readouterr().out
    make_heads(app / "heads")
    (app / "heads" / heads.PLACEHOLDER).write_text("CI")
    assert entry.main(["run", "--input", "x", "--output", "window"]) == entry.NO_HEADS_EXIT
    assert "HEADS_PLACEHOLDER" in capsys.readouterr().out


def test_run_is_refused_on_builtin_graphics(product_env, monkeypatch, capsys, tmp_path):
    from streamsafe import syscheck

    monkeypatch.setattr(updates, "expired", lambda today=None, expires=None: False)
    monkeypatch.setattr(heads, "why_not", lambda env: "")
    monkeypatch.setattr(syscheck, "_gpus", lambda: [syscheck.Gpu("Intel(R) UHD Graphics 630", "intel", 1.0)])
    monkeypatch.delenv("STREAMSAFE_SKIP_GPU_CHECK", raising=False)
    import cv2
    import numpy as np

    clip = str(tmp_path / "clip.mp4")  # the input opens first (OBS can connect while the slower checks run)
    w = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 36))
    for _ in range(3):
        w.write(np.zeros((36, 64, 3), np.uint8))
    w.release()
    assert entry.main(["run", "--input", clip, "--output", "window", "--report-dir", str(tmp_path)]) == entry.WEAK_GPU_EXIT == 6
    out = capsys.readouterr().out
    assert "WEAK_GPU" in out and "Nothing was streamed" in out


def test_about_says_when_the_beta_ends(product_env, capsys, monkeypatch):
    from streamsafe.app import build_info

    monkeypatch.setattr(build_info, "EXPIRES", "2999-01-01")
    assert entry.main(["about", "--json"]) == 0
    facts = json.loads(capsys.readouterr().out)
    assert facts["expires"] == "2999-01-01" and facts["expired"] is False and facts["days_left"] > 1000
    assert "ProgramData" not in facts["data_dir"]


def test_unknown_command_prints_usage(capsys):
    assert entry.main(["nonsense"]) == entry.USAGE_EXIT
    assert "check" in capsys.readouterr().out


# --- texts ------------------------------------------------------------------------------------------------------
def used_keys() -> set[str]:
    keys = set()
    for f in ("ui.py", "controller.py"):
        text = (ROOT / "streamsafe" / "app" / f).read_text(encoding="utf-8")
        keys |= set(re.findall(r"""\btr\(\s*["']([a-z_0-9]+)["']""", text))
        keys |= set(re.findall(r'"((?:why|bad|engine|st|beta|verdict|src|tab|chk|p)_[a-z_0-9]+)"', text))
    return keys


def test_every_text_exists_in_both_languages():
    assert i18n.missing() == []
    unknown = {
        k for k in used_keys() if k not in i18n.TEXT and not k.startswith(("chk_", "p_", "st_hint", "src_", "plat_", "why_", "plan_"))
    }
    assert unknown == set()
    for k in ("src_private", "src_missing", "src_network", "src_other"):
        assert k in i18n.TEXT
    for k in (
        "chk_os",
        "chk_python",
        "chk_gpu",
        "chk_runtime",
        "chk_ram",
        "chk_disk",
        "chk_ffmpeg",
        "chk_obs",
        "chk_models",
        "chk_engines",
    ):
        assert k in i18n.TEXT


def test_arabic_texts_are_arabic_and_names_come_from_branding():
    from streamsafe.branding import PRODUCT_NAME

    for key in ("st_protected", "st_covered", "start", "stop", "sec_check", "beta_expired", "src_private"):
        assert re.search("[؀-ۿ]", i18n.TEXT[key][0]), key
        assert not re.search("[؀-ۿ]", i18n.TEXT[key][1]), key
    assert i18n.t("ar", "st_protected") == "محمي" and i18n.t("en", "st_protected") == "Protected"
    assert i18n.t("ar", "st_covered") == "مغطى بالكامل" and i18n.t("en", "st_covered") == "Fully covered"
    assert i18n.t("en", "about", version="1.0") == f"{PRODUCT_NAME} 1.0" and i18n.t("en", "no_such_key") == "no_such_key"
    assert i18n.t("en", "beta_days", days=3) != ""  # a missing field never raises


def test_product_name_is_in_one_place():
    hits = []
    for p in (ROOT / "streamsafe").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"(?<![\w.])StreamSafe(?![\w.])", text):
            line = text[: m.start()].count("\n") + 1
            hits.append((p.name, line))
    allowed = {"branding.py", "i18n.py"}  # i18n uses {name}; docstrings elsewhere are not user text
    user_text = [h for h in hits if h[0] not in allowed and h[0] in ("ui.py", "controller.py")]
    assert user_text == []


# --- build tools ------------------------------------------------------------------------------------------------
def test_file_version_and_resource_template():
    from tools import build_meta

    assert build_meta.file_version("3.0.0", 57) == (3, 0, 0, 57) and build_meta.file_version("3.1", 70000) == (3, 1, 0, 65535)
    text = build_meta.VERSION_TEMPLATE.format(
        a=3, b=0, c=0, d=57, company="C", description="D", version="3.0.0.57", internal="I", copyright="(c)", original="I.exe", product="P"
    )
    assert "filevers=(3, 0, 0, 57)" in text and "StringStruct('ProductName', 'P')" in text
    import ast

    ast.parse(text)  # the resource file is valid Python for PyInstaller


def test_build_meta_stamps_a_future_date_and_refuses_a_past_one(tmp_path):
    keep = (ROOT / "streamsafe" / "app" / "build_info.py").read_text(encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_meta.py"), "--days", "30", "--build", "9", "--out", str(tmp_path)],
                           capture_output=True, text=True, check=False)  # fmt: skip
        assert r.returncode == 0, r.stderr
        facts = dict(line.split("=", 1) for line in r.stdout.splitlines())
        stamp = (ROOT / "streamsafe" / "app" / "build_info.py").read_text(encoding="utf-8")
        import datetime as dt

        assert f'EXPIRES = "{facts["expires"]}"' in stamp and 'VERSION = "3.0.0.9"' in stamp
        assert (dt.date.fromisoformat(facts["expires"]) - dt.date.today()).days == 30
        assert (tmp_path / "version_StreamSafe.txt").is_file() and (tmp_path / "version_StreamSafeApp.txt").is_file()
        bad = subprocess.run([sys.executable, str(ROOT / "tools" / "build_meta.py"), "--expires", "2020-01-01", "--out", str(tmp_path)],
                             capture_output=True, text=True, check=False)  # fmt: skip
        assert bad.returncode == 2
    finally:
        (ROOT / "streamsafe" / "app" / "build_info.py").write_text(keep, encoding="utf-8")


def test_ffmpeg_lock_is_pinned_lgpl_and_unpack_keeps_only_what_is_needed(tmp_path):
    from tools import get_ffmpeg

    lock = json.loads(get_ffmpeg.LOCK.read_text(encoding="utf-8"))
    assert lock["url"].startswith("https://") and re.fullmatch(r"[0-9a-f]{64}", lock["sha256"])
    assert "lgpl" in lock["name"] and "-gpl" not in lock["name"].replace("lgpl", "")
    z = tmp_path / "f.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for n in ("bin/ffmpeg.exe", "bin/avcodec-62.dll", "bin/ffplay.exe", "include/x.h", "LICENSE.txt"):
            zf.writestr("ffmpeg-n8/" + n, "x")
    got = get_ffmpeg.unpack(z, tmp_path / "out", lock["keep"])
    assert sorted(got) == ["LICENSE.txt", "bin/avcodec-62.dll", "bin/ffmpeg.exe"]
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "get_ffmpeg.py"), str(tmp_path / "o2"), "--zip", str(z)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 1 and "sha256 mismatch" in r.stderr and z.exists()  # a wrong archive is refused and not deleted


def test_make_heads_placeholder_is_marked_and_real_heads_are_required_for_release(tmp_path):
    from tools import make_heads

    assert make_heads.main([str(tmp_path / "t")]) == 0
    out = tmp_path / "t" / "heads"
    assert heads.problem(out) == "placeholder" and (out / heads.PLACEHOLDER).is_file()
    real = ROOT / "installer" / "heads"
    if not (real / "fast_h6.txt").exists():
        assert make_heads.main([str(tmp_path / "r"), "--require-real"]) == 1
