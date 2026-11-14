"""System checker: can this PC protect a live stream? ``python -m streamsafe.syscheck [--models DIR] [--engines] [--json]``.

Run by the installer before the first start and by the app's "check my PC" button. Every check gives
ok / warn / fail plus a plain-language fix in Arabic and English, so a streamer (not a programmer) knows what to do.
Any "fail" means the engine will refuse to start (fail closed) - better to say so here than mid-stream.

Thresholds marked "estimate" are our current best guesses, to be replaced by the consumer-GPU measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from streamsafe.branding import PRODUCT_NAME

OK, WARN, FAIL = "ok", "warn", "fail"
MIN_PYTHON = (3, 11)
FAIL_VRAM_GB = 4.0  # below this the models do not fit: on shared memory the whole PC can freeze (seen on an Intel UHD 630, 1 GB)
MIN_VRAM_GB = 6.0  # estimate: fp32 eye 2.6 GB + detector/pose/scene + frames; fp16 eye roughly halves the eye
MIN_RAM_GB = 8.0  # estimate: the delay buffer + models on the host side
MIN_DISK_GB = 2.0  # reports, logs and the ONNX Runtime engine cache
OBS_PATHS = (r"C:\Program Files\obs-studio\bin\64bit\obs64.exe", r"C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe")


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix_ar: str = ""
    fix_en: str = ""


@dataclass
class Gpu:
    name: str
    vendor: str  # nvidia / amd / intel / other
    vram_gb: float | None = None  # None: the OS did not say (Windows reports at most 4 GB through WMI)
    driver: str = ""


@dataclass
class Probes:
    """Everything the checker reads from the machine, injectable so the rules are testable without the hardware."""

    system: Callable[[], str] = platform.system
    python: Callable[[], tuple[int, int]] = lambda: sys.version_info[:2]
    ort_providers: Callable[[], list[str] | None] = field(default_factory=lambda: _ort_providers)
    gpus: Callable[[], list[Gpu]] = field(default_factory=lambda: _gpus)
    ram_gb: Callable[[], float | None] = field(default_factory=lambda: _ram_gb)
    disk_free_gb: Callable[[str], float] = lambda p: shutil.disk_usage(p).free / 1e9
    which: Callable[[str], str | None] = shutil.which
    exists: Callable[[str], bool] = os.path.exists


def vendor_of(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or n.startswith("rtx"):
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n or " arc" in f" {n}":
        return "intel"
    return "other"


_IGPU_HINTS = (
    "uhd graphics",
    "hd graphics",
    "iris",
    "radeon(tm) graphics",
    "radeon graphics",
    "radeon vega",
    "vega 8",
    "vega 11",
    "radeon 610m",
    "radeon 660m",
    "radeon 680m",
    "radeon 740m",
    "radeon 760m",
    "radeon 780m",
    "radeon 880m",
    "radeon 890m",
)


def is_integrated(gpu: Gpu) -> bool:
    """A graphics chip built into the processor (shares system memory, far too slow for the models)."""
    n = gpu.name.lower()
    if gpu.vendor == "nvidia" or " arc" in f" {n}":
        return False
    return any(h in n for h in _IGPU_HINTS)


def _ort_providers() -> list[str] | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    return list(ort.get_available_providers())


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def nvidia_gpus() -> list[Gpu]:
    """Every NVIDIA card with its driver version, as the driver's own tool reports them (nvidia-smi is installed
    with the driver). Also what a TensorRT engine is tied to (``models/onnx_backend/engines.py``)."""
    found: list[Gpu] = []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            mem = float(parts[1]) / 1024 if parts[1].replace(".", "", 1).isdigit() else None
            found.append(Gpu(parts[0], "nvidia", mem, parts[2]))
    return found


def _gpus() -> list[Gpu]:
    found = nvidia_gpus()
    if platform.system() == "Windows":
        ps = "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.AdapterRAM + '|' + $_.DriverVersion }"
        for line in _run(["powershell", "-NoProfile", "-Command", ps]).strip().splitlines():
            name, _, rest = line.partition("|")
            ram, _, driver = rest.partition("|")
            name = name.strip()
            if not name or any(g.name == name for g in found) or "basic display" in name.lower():
                continue
            vram = int(ram) / 2**30 if ram.strip().isdigit() and int(ram) > 0 else None
            # WMI's AdapterRAM is a 32-bit field: 4 GB or more reads as ~4 GB, so treat it as unknown
            found.append(Gpu(name, vendor_of(name), None if vram is not None and vram >= 3.9 else vram, driver.strip()))
    return found


def _ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "ave")
                ]

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return st.total / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, ValueError, OSError):
        return None


def check_python(p: Probes) -> Check:
    v = p.python()
    if v >= MIN_PYTHON:
        return Check("python", OK, f"Python {v[0]}.{v[1]}")
    return Check(
        "python",
        FAIL,
        f"Python {v[0]}.{v[1]} is too old",
        "أعد تثبيت البرنامج من المثبّت الرسمي.",
        f"Reinstall {PRODUCT_NAME} with the official installer.",
    )


def check_os(p: Probes) -> Check:
    s = p.system()
    if s == "Windows":
        return Check("os", OK, "Windows")
    return Check(
        "os",
        WARN,
        f"{s}: only Windows is supported for live streaming",
        "البث المباشر مدعوم على ويندوز فقط.",
        "Live streaming is supported on Windows only.",
    )


def check_gpu(p: Probes) -> tuple[Check, Gpu | None]:
    gpus = p.gpus()
    if not gpus:
        fix_ar = "ما لقينا كرت شاشة. ثبّت آخر تعريف لكرتك (NVIDIA أو AMD أو Intel) ثم أعد الفحص."
        return Check(
            "gpu", FAIL, "no graphics card found", fix_ar, "No graphics card found. Install the latest driver, then check again."
        ), None
    rank = {"nvidia": 0, "amd": 1, "intel": 2, "other": 3}
    best = min(gpus, key=lambda g: (rank[g.vendor], -(g.vram_gb or 0)))
    names = ", ".join(g.name for g in gpus)
    virtual = best.vendor == "other"  # Hyper-V / VMware / remote-desktop adapters: no real graphics card
    if virtual or is_integrated(best) or (best.vram_gb is not None and best.vram_gb < FAIL_VRAM_GB):
        what = "no real graphics card" if virtual else "built-in graphics" if is_integrated(best) else f"{best.vram_gb:.1f} GB video memory"
        return (
            Check(
                "gpu",
                FAIL,
                f"{best.name}: {what}; {PRODUCT_NAME} needs a gaming graphics card (NVIDIA RTX 3060 class, 6 GB+)",
                "كرت الشاشة في جهازك ضعيف (كرت مدمج أو ذاكرته قليلة)، والبرنامج يحتاج كرت ألعاب مثل NVIDIA RTX 3060 أو أعلى. "
                "التشغيل ممنوع عشان ما يتعلق جهازك.",
                f"This graphics card is too weak (built-in or low memory). {PRODUCT_NAME} needs a gaming card such as an "
                "NVIDIA RTX 3060 or better. Starting is blocked so your PC does not freeze.",
            ),
            best,
        )
    if best.vram_gb is not None and best.vram_gb < MIN_VRAM_GB:
        return (
            Check(
                "gpu",
                WARN,
                f"{best.name}: {best.vram_gb:.1f} GB video memory (we suggest {MIN_VRAM_GB:.0f} GB+)",
                "ذاكرة كرت الشاشة قليلة. سكّر البرامج الثانية اللي تستخدم الكرت وقت البث؛ ممكن تطلع شاشة التغطية الكاملة أكثر.",
                "Low video memory: close other apps that use the graphics card while streaming; the full-screen cover may show more often.",
            ),
            best,
        )
    mem = f", {best.vram_gb:.0f} GB" if best.vram_gb else ""
    return Check("gpu", OK, f"{best.name} ({best.vendor}{mem}); all: {names}"), best


def check_runtime(p: Probes, gpu: Gpu | None) -> Check:
    providers = p.ort_providers()
    if providers is None:
        return Check(
            "runtime",
            FAIL,
            "ONNX Runtime is not installed",
            "الملفات ناقصة. أعد تثبيت البرنامج.",
            f"Files are missing. Reinstall {PRODUCT_NAME}.",
        )
    if "CUDAExecutionProvider" in providers:
        return Check("runtime", OK, "NVIDIA CUDA")
    if "DmlExecutionProvider" in providers:
        if gpu is not None and gpu.vendor == "nvidia":
            return Check(
                "runtime",
                WARN,
                "DirectML on an NVIDIA card (works, slower than the NVIDIA package)",
                "عندك كرت إنفيديا: ثبّت «الوضع السريع لإنفيديا» من نافذة البرنامج عشان يكون أسرع.",
                "You have an NVIDIA card: install the fast NVIDIA mode from the app window for more speed.",
            )
        return Check("runtime", OK, "DirectML (DirectX 12)")
    return Check(
        "runtime",
        FAIL,
        f"no GPU runtime (have: {providers})",
        "البرنامج ما يقدر يستخدم كرت الشاشة. حدّث تعريف الكرت وويندوز ثم أعد الفحص.",
        f"{PRODUCT_NAME} cannot use your graphics card. Update the driver and Windows, then check again.",
    )


def check_ram(p: Probes) -> Check:
    gb = p.ram_gb()
    if gb is None:
        return Check("ram", WARN, "could not read the memory size")
    if gb < MIN_RAM_GB:
        return Check(
            "ram", WARN, f"{gb:.0f} GB RAM", "الرام قليلة. سكّر البرامج الثانية وقت البث.", "Low RAM: close other apps while streaming."
        )
    return Check("ram", OK, f"{gb:.0f} GB RAM")


def check_disk(p: Probes, path: str) -> Check:
    try:
        gb = p.disk_free_gb(path)
    except OSError:
        return Check("disk", WARN, f"could not read free space at {path}")
    if gb < MIN_DISK_GB:
        return Check("disk", WARN, f"{gb:.1f} GB free at {path}", "المساحة قليلة. فضّي مساحة على القرص.", "Low disk space: free some space.")
    return Check("disk", OK, f"{gb:.0f} GB free")


def check_ffmpeg(p: Probes) -> Check:
    if p.which("ffmpeg"):
        return Check("ffmpeg", OK, "FFmpeg found")
    return Check(
        "ffmpeg", FAIL, "FFmpeg not found", "ملف FFmpeg ناقص. أعد تثبيت البرنامج.", f"FFmpeg is missing. Reinstall {PRODUCT_NAME}."
    )


def check_obs(p: Probes) -> Check:
    if p.system() != "Windows":
        return Check("obs", WARN, "OBS check runs on Windows only")
    if any(p.exists(path) for path in OBS_PATHS):
        return Check("obs", OK, "OBS Studio found")
    return Check(
        "obs",
        WARN,
        "OBS Studio not found in its usual folder",
        "ما لقينا OBS. ثبّته أو تأكد من مكانه.",
        "OBS Studio not found: install it or check its folder.",
    )


def check_models(folder: str, quick: bool) -> Check:
    """The model folder against its manifest. ``quick`` checks sizes only (hashing the eye takes a while on a HDD)."""
    from streamsafe.models.onnx_backend import REQUIRED, verify

    path = Path(folder)
    if not (path / "manifest.json").is_file():
        return Check(
            "models",
            FAIL,
            "models not installed yet / النماذج غير منزّلة بعد",
            "نزّل النماذج من الصفحة الرئيسية (مرة وحدة).",
            "Download the models from the Home tab (once).",
        )
    fix_ar = "ملفات النماذج ناقصة أو تالفة. افتح نافذة البرنامج ونزّل النماذج من جديد (أو اختر ملف النماذج اللي أرسلناه لك)."
    fix_en = "Model files are missing or damaged. Open the app window and get the models again (or pick the model file we sent you)."
    try:
        if quick:
            files = json.loads((path / "manifest.json").read_text())["files"]
            bad = [n for n in REQUIRED if n not in files]
            bad += [n for n, m in files.items() if not (path / n).exists() or (path / n).stat().st_size != m["bytes"]]
            if bad:
                return Check("models", FAIL, f"missing or damaged: {bad}", fix_ar, fix_en)
        else:
            verify(path, {})  # pins are checked again by the engine itself at start
    except (OSError, ValueError, KeyError) as e:
        return Check("models", FAIL, str(e), fix_ar, fix_en)
    return Check("models", OK, f"models in {folder}" + (" (sizes only)" if quick else " (sha256 verified)"))


def check_engines(folder: str, quick: bool) -> Check:
    """Optional (``--engines``, NVIDIA edition): is there a TensorRT engine for every model, built for this card,
    this driver and these libraries? "No" is a warning, not a failure: the app prepares them before the stream
    starts (``python -m streamsafe.prepare``, minutes), and without them the engine runs on CUDA and says so."""
    from streamsafe import prepare
    from streamsafe.config import Config

    fix_ar = "تجهيز كرت الشاشة ما اكتمل. البرنامج يجهّزه قبل أول بث (يأخذ دقائق، مرة واحدة)."
    fix_en = f"The graphics card is not prepared yet. {PRODUCT_NAME} does it before the first stream (a few minutes, once)."
    try:
        rows = prepare.status(Config.from_env(onnx_dir=folder, trt=True), quick)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        return Check(
            "engines",
            WARN,
            f"TensorRT cannot be used: {e}",
            "الوضع السريع لكروت إنفيديا غير متاح، والبرنامج يشتغل على الوضع الأبطأ. "
            "حدّث تعريف الكرت أو أعد تثبيت الوضع السريع من نافذة البرنامج.",
            f"The fast NVIDIA mode is not available; {PRODUCT_NAME} runs in the slower mode. "
            "Update the driver or install the fast mode again from the app window.",
        )
    waiting = [f"{label} ({why})" for label, _, why in rows if why]
    if waiting:
        return Check("engines", WARN, f"TensorRT engines not ready: {'; '.join(waiting)}", fix_ar, fix_en)
    return Check("engines", OK, f"TensorRT engines ready ({len(rows)}/{len(rows)})" + (" (sizes only)" if quick else ""))


def run_checks(probes: Probes | None = None, models: str = "", quick: bool = False, engines: bool = False) -> list[Check]:
    p = probes or Probes()
    gpu_check, gpu = check_gpu(p)
    checks = [check_os(p), check_python(p), gpu_check, check_runtime(p, gpu), check_ram(p)]
    checks += [check_disk(p, _data_disk(p)), check_ffmpeg(p), check_obs(p)]
    if models:
        checks.append(check_models(models, quick))
        if engines:
            checks.append(check_engines(models, quick))
    return checks


def _data_disk(p: Probes) -> str:
    """Where models, engines and reports are written: the user's data folder, or the nearest folder above it
    that exists already (the first start creates the rest)."""
    from streamsafe.app import paths

    folder = paths.data_dir()
    for candidate in (folder, *folder.parents):
        if p.exists(str(candidate)):
            return str(candidate)
    return os.getcwd()


def verdict(checks: list[Check]) -> str:
    if any(c.status == FAIL for c in checks):
        return FAIL
    return WARN if any(c.status == WARN for c in checks) else OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="streamsafe.syscheck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.environ.get("ONNX_DIR", ""), help="ONNX model folder to verify")
    ap.add_argument("--quick", action="store_true", help="check model sizes only, not sha256")
    ap.add_argument("--engines", action="store_true", help="with --models: are the TensorRT engines ready (NVIDIA edition)?")
    ap.add_argument("--json", action="store_true", help="machine-readable output for the installer / app")
    ap.add_argument("--lang", choices=["ar", "en"], default="en")
    args = ap.parse_args(argv)
    checks = run_checks(models=args.models, quick=args.quick, engines=args.engines)
    result = verdict(checks)
    if args.json:
        print(json.dumps({"verdict": result, "checks": [asdict(c) for c in checks]}, ensure_ascii=False, indent=1))
    else:
        for c in checks:
            fix = c.fix_ar if args.lang == "ar" else c.fix_en
            print(f"[{c.status.upper():4}] {c.name:8} {c.detail}" + (f"\n         -> {fix}" if fix and c.status != OK else ""))
        print(f"verdict: {result}")
    return {OK: 0, WARN: 1, FAIL: 2}[result]


if __name__ == "__main__":
    sys.exit(main())
