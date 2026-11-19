"""Edits the tutorial: the Windows screen recording (tools/tutorial/drive_app.py) + the GPU pod's live copy of what
viewers got on Kick (tools/demo_live_job.sh) -> one video with Arabic step captions, zooms and the key blurred.

    python tools/tutorial/edit.py SCREEN.mp4 ACTIONS.json VIEWER.mkv OUT_DIR

Captions are drawn by Chromium (Playwright), so the Arabic is shaped right; FFmpeg cuts, zooms and joins.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

SCREEN, ACTIONS, VIEWER, OUT = (Path(a) for a in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
W, H, FPS = 1920, 1080, 30
ROOT = Path(__file__).resolve().parents[2]
LOGO = base64.b64encode((ROOT / "marketing" / "logo_rgba.png").read_bytes()).decode()
acts = {a["step"]: a for a in json.loads(ACTIONS.read_text(encoding="utf-8"))}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@500;700&display=swap');
html,body{margin:0;width:1920px;height:1080px;background:transparent;font-family:'IBM Plex Sans Arabic',sans-serif;direction:rtl}
.bar{position:absolute;right:60px;left:60px;bottom:56px;display:flex;align-items:center;gap:28px;padding:26px 36px;
 background:rgba(18,11,43,.92);border-radius:22px;box-shadow:0 10px 40px rgba(0,0,0,.35);border-right:10px solid #ff4fa3}
.num{flex:none;width:84px;height:84px;border-radius:50%;background:#ff4fa3;color:#fff;font-weight:700;font-size:46px;
 display:grid;place-items:center}
.txt{color:#fff;font-size:44px;font-weight:700;line-height:1.35}
.sub{display:block;color:#cbbfe6;font-size:30px;font-weight:500;margin-top:6px}
.tag{position:absolute;top:44px;right:60px;background:#e0182f;color:#fff;font-weight:700;font-size:30px;
 padding:10px 22px;border-radius:12px}
.card{position:absolute;inset:0;background:radial-gradient(70% 70% at 50% 30%,#2a1a5e,#120b2b);display:flex;flex-direction:column;
 align-items:center;justify-content:center;color:#fff;text-align:center;gap:24px}
.card img{width:260px}.card h1{font-size:78px;margin:0}.card p{font-size:40px;margin:0;color:#e6dcff}
.card .pill{background:#ff4fa3;color:#fff;font-weight:700;font-size:40px;padding:14px 34px;border-radius:999px;direction:ltr}
"""


def page(body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def caption(num: str, text: str, sub: str = "", tag: str = "") -> str:
    n = f"<div class='num'>{num}</div>" if num else ""
    s = f"<span class='sub'>{sub}</span>" if sub else ""
    t = f"<div class='tag'>{tag}</div>" if tag else ""
    return page(f"{t}<div class='bar'>{n}<div class='txt'>{text}{s}</div></div>")


def card(title: str, line: str, pill: str = "") -> str:
    p = f"<div class='pill'>{pill}</div>" if pill else ""
    return page(f"<div class='card'><img src='data:image/png;base64,{LOGO}'><h1>{title}</h1><p>{line}</p>{p}</div>")


def render(htmls: dict[str, str]) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H})
        for name, html in htmls.items():
            pg.set_content(html, wait_until="networkidle")
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(OUT / f"{name}.png"), omit_background=True)
        b.close()


def t(step: str) -> float:
    return float(acts[step]["t"])


def zoom_box(step: str, w: int = 960, h: int = 540) -> tuple[int, int, int, int]:
    """A 16:9 crop of the screen centred on a step's widget (kept inside the screen)."""
    x, y, bw, bh = acts[step]["box"]
    cx, cy = x + bw // 2, y + bh // 2
    left = min(max(cx - w // 2, 0), W - w)
    top = min(max(cy - h // 2, 0), H - h)
    return left, top, w, h


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]
parts: list[Path] = []


def still(name: str, png: str, dur: float) -> None:
    out = OUT / f"p_{name}.mp4"
    vf = f"fade=in:0:12,fade=out:st={dur - 0.4}:d=0.4,format=yuv420p"
    ff("-loop", "1", "-t", f"{dur}", "-i", str(OUT / png), "-vf", vf, *ENC, str(out))
    parts.append(out)


def screen(name: str, start: float, end: float, cap: str, box: tuple[int, int, int, int] | None = None,
           blur: tuple[int, int, int, int] | None = None, speed: float = 1.0, src: Path = SCREEN) -> None:  # fmt: skip
    """A piece of a recording: optional zoom (crop -> 1920x1080), a blurred box (screen coords), the caption."""
    out = OUT / f"p_{name}.mp4"
    chain = []
    if blur is not None:
        bx, by, bw, bh = blur
        chain.append(f"[0:v]split[a][b];[b]crop={bw}:{bh}:{bx}:{by},gblur=sigma=9[k];[a][k]overlay={bx}:{by}[v0]")
    else:
        chain.append("[0:v]null[v0]")
    if box is not None:
        x, y, w, h = box
        chain.append(f"[v0]crop={w}:{h}:{x}:{y},scale={W}:{H}:flags=lanczos[v1]")
    else:
        chain.append(f"[v0]scale={W}:{H}[v1]")
    chain.append(f"[v1]setpts=PTS/{speed}[v2];[v2][1:v]overlay=0:0,fade=in:0:8[v]")
    ff("-ss", f"{max(start, 0):.2f}", "-to", f"{end:.2f}", "-i", str(src), "-i", str(OUT / cap),
       "-filter_complex", ";".join(chain), "-map", "[v]", *ENC, str(out))  # fmt: skip
    parts.append(out)


def main() -> None:
    render({
        "c_title": card("طريقة تشغيل StreamSafe", "من أول ما تفتحه لين بثك يطلع محمي"),
        "c_open": caption("", "افتح StreamSafe", "أول مرة بس: اضغط «نزّل النماذج» وانتظر لين يخلص"),
        "c_s1": caption("1", "اختر منصتك", "كيك، تويتش، يوتيوب، تيك توك..."),
        "c_s2": caption("2", "الصق مفتاح البث", "تاخذه من لوحة التحكم في منصتك (هنا مغبّش)"),
        "c_s3": caption("3", "سكّر OBS واضغط «اضبط OBS تلقائياً»", "مرة وحدة بس. البرنامج يغيّر عنوان السيرفر في OBS وبس"),
        "c_go": caption("▶", "اضغط «ابدأ الحماية»", "البرنامج يفتح OBS بنفسه، وOBS يبدأ يرسل له البث"),
        "c_live": caption("", "هذا اللي يشوفه جمهورك", "البث متأخر ثانيتين، وأي مشهد مكشوف يتغطى قبل ما يوصلهم", tag="● بث حقيقي على كيك"),
        "c_stop": caption("■", "خلصت؟ اضغط «إيقاف»", "ومن بكرة: افتح البرنامج واضغط «ابدأ الحماية» بس"),
        "c_out": card("حمّله مجاناً", "streamsafe-sa.vercel.app", "ko-fi.com/streamsafe ♥"),
    })  # fmt: skip
    still("title", "c_title.png", 3.5)
    screen("open", t("app_open") - 0.5, t("step1_platform"), "c_open.png", box=zoom_box("app_open", 1280, 720))
    screen("s1", t("step1_platform") - 0.3, t("step1_done") + 0.6, "c_s1.png", box=zoom_box("step1_platform"))
    kx, ky, kw, kh = acts["step2_key"]["box"]
    screen("s2", t("step2_key") - 0.2, t("step2_done") + 0.8, "c_s2.png", box=zoom_box("step2_key"), blur=(kx, ky, kw, kh))
    screen("s3", t("step3_obs") - 0.2, t("step3_done") + 1.8, "c_s3.png", box=zoom_box("step3_obs"))
    go_end = min(t("running") + 1, t("start") + 40)
    screen("go", t("start") - 0.4, go_end, "c_go.png", speed=max(1.0, (go_end - t("start")) / 16))
    screen("live", 42, 70, "c_live.png", src=VIEWER)
    screen("stop", t("stop") - 0.8, t("end"), "c_stop.png", box=zoom_box("stop", 1280, 720))
    still("out", "c_out.png", 4.5)
    lst = OUT / "parts.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    final = OUT / "StreamSafe_tutorial.mp4"
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
       "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(final))  # fmt: skip
    for i, s in enumerate((5, 15, 25, 40, 60, 80, 100)):
        ff("-ss", str(s), "-i", str(final), "-frames:v", "1", "-vf", "scale=960:-1", str(OUT / f"check_{i}.jpg"))
    print("FINAL", final, final.stat().st_size, flush=True)


main()
