"""Entry point of the installed program (and ``python -m streamsafe.app.entry`` from source).

    <name>.exe check [--json]          system check (GPU, runtime, FFmpeg, OBS, models)
    <name>.exe prepare                 build / verify the per-PC engines (NVIDIA TensorRT)
    <name>.exe run --input ... --output ...   the stream filter (same options as python -m streamsafe)
    <name>.exe gpu-probe               which device ONNX Runtime really reaches (one JSON line)
    <name>.exe about [--json]          version, build date, beta expiry, folders in use
    <name>.exe encoder-probe           which H.264 encoder of the bundled FFmpeg works on this PC
    <name>.exe ui                      the window (what <name>App.exe opens)

In the installed program the product settings are filled in where the environment does not set them
(``product_defaults``): the ONNX backend, the model folder ``<data>/models`` (or ``--models DIR``), the heads,
the OBS state file, the engine cache, and the NVIDIA runtime with TensorRT when its libraries are installed.
``run --output env:NAME`` takes the output address from the environment variable NAME, so a stream key never has
to be on this program's command line. A beta build past its date refuses ``run`` (exit code 4), and so does a
build without usable risk heads (5): it never streams, so it can never stream uncovered.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

from streamsafe.app import paths, runtime_select
from streamsafe.branding import CLI_EXE, GUI_EXE, PRODUCT_NAME

EXPIRED_EXIT = 4
NO_HEADS_EXIT = 5
USAGE_EXIT = 2


def product_defaults(env: MutableMapping[str, str], models: str = "") -> None:
    """Fill in the product's runtime settings where the environment (or the app that started us) set none."""
    from streamsafe.app import heads, nvidia_pack

    if models:
        env["ONNX_DIR"] = models
    env.setdefault("BACKEND", "onnx")
    env.setdefault("ONNX_DIR", str(paths.models_dir(env)))
    if "HEAD_DIR" not in env:  # the newer of the installed heads and the updated ones; none usable: the engine fails closed
        chosen = heads.best(env)
        env["HEAD_DIR"] = str(chosen or paths.bundled_heads_dir())
    env.setdefault("OBS_STATE", str(paths.obs_state_file(env)))
    env.setdefault("TRT_CACHE", str(paths.trt_cache_dir(env)))
    if runtime_select.RUNTIME_ENV not in env and nvidia_pack.ready(paths.nvidia_dir(env)):
        env[runtime_select.RUNTIME_ENV] = runtime_select.NVIDIA
    if runtime_select.wanted(env) == runtime_select.NVIDIA:
        env.setdefault("TRT", "1")
        env.setdefault("TRT_BUILD", "0")  # never minutes of engine building at "go live": ``prepare`` does that


def take_option(args: list[str], name: str) -> tuple[str, list[str]]:
    """Remove ``name VALUE`` / ``name=VALUE`` from ``args``; returns (VALUE or "", the other arguments)."""
    value, rest, i = "", [], 0
    while i < len(args):
        if args[i] == name and i + 1 < len(args):
            value, i = args[i + 1], i + 2
        elif args[i].startswith(name + "="):
            value, i = args[i].split("=", 1)[1], i + 1
        else:
            rest.append(args[i])
            i += 1
    return value, rest


def resolve_env_outputs(args: list[str], env: MutableMapping[str, str]) -> list[str]:
    """``--output env:NAME`` -> ``--output <value of NAME>``. A missing variable is an error, not an empty output:
    a run that silently had no output would look like a stream."""
    out = list(args)
    for i, arg in enumerate(out):
        spec = arg.split("=", 1)[1] if arg.startswith("--output=") else arg if i and out[i - 1] == "--output" else None
        if spec is not None and spec.startswith("env:"):
            name = spec[4:]
            if not env.get(name):
                raise SystemExit(f"--output env:{name}: the environment variable {name} is not set")
            out[i] = arg[: len(arg) - len(spec)] + env[name]
    return out


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a code page that cannot print Arabic
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _is_window_exe() -> bool:
    return paths.is_frozen() and Path(sys.executable).stem.lower() == GUI_EXE.lower()


def about(as_json: bool) -> int:
    import json

    from streamsafe.app import build_info, updates

    facts = {
        "product": PRODUCT_NAME,
        "version": updates.current_version(),
        "build_date": build_info.BUILD_DATE,
        "expires": build_info.EXPIRES,
        "days_left": updates.days_left(),
        "expired": updates.expired(),
        "frozen": paths.is_frozen(),
        "program_dir": str(paths.app_dir()),
        "data_dir": str(paths.data_dir()),
        "ffmpeg": paths.find_ffmpeg(),
    }
    print(json.dumps(facts, ensure_ascii=False) if as_json else "\n".join(f"{k}: {v}" for k, v in facts.items()))
    return 0


def encoder_probe() -> int:
    import json

    from streamsafe.app import encoders

    ffmpeg = paths.find_ffmpeg()
    listed = sorted(encoders.listed(ffmpeg)) if ffmpeg else []
    chosen = encoders.choose(ffmpeg) if ffmpeg else ""
    print("ENCODER_PROBE " + json.dumps({"ffmpeg": ffmpeg, "listed": listed, "chosen": chosen}), flush=True)
    return 0 if chosen else 1


def main(argv: list[str] | None = None) -> int:
    _utf8_streams()
    # the analyzer runs in a child process and a frozen exe re-enters here for it: the runtime choice (inherited
    # through the environment) must be in effect before that child imports anything
    runtime_select.activate()
    multiprocessing.freeze_support()
    argv = sys.argv[1:] if argv is None else list(argv)
    cmd, rest = (argv[0], argv[1:]) if argv else ("ui" if _is_window_exe() else "", [])
    paths.use_bundled_ffmpeg()
    product = paths.is_frozen() or os.environ.get("STREAMSAFE_PRODUCT") == "1"
    if cmd == "ui":
        from streamsafe.app.ui import main as m

        return m(rest)
    if cmd in ("check", "prepare", "run", "gpu-probe") and product:
        models, rest = take_option(rest, "--models")
        product_defaults(os.environ, models)
        why = runtime_select.activate()
        if runtime_select.wanted() == runtime_select.NVIDIA and why:
            print(f"STREAMSAFE_RUNTIME nvidia not used: {why}", file=sys.stderr, flush=True)
            if cmd == "run":  # asked for and not there: say so and stop, never start on an unknown runtime
                return USAGE_EXIT
    if cmd == "about":
        return about("--json" in rest)
    if cmd == "encoder-probe":
        return encoder_probe()
    if cmd == "translate-selftest":  # CI: the packaged program can load the translation libraries and draw Arabic
        return translate_selftest()
    if cmd == "check":
        from streamsafe.syscheck import main as m

        if product:
            rest = [*rest, "--models", os.environ["ONNX_DIR"]]
            if runtime_select.active() and "--engines" not in rest:
                rest.append("--engines")
    elif cmd == "prepare":
        from streamsafe.prepare import main as m
    elif cmd == "gpu-probe":
        from streamsafe.app.gpu_probe import main as m
    elif cmd == "run":
        from streamsafe.app import encoders, engine, heads, updates

        if updates.expired():
            print(f"BETA_EXPIRED this beta build of {PRODUCT_NAME} ended; install the new version. Nothing was streamed.", flush=True)
            return EXPIRED_EXIT
        if product and (why := heads.why_not(os.environ)):
            print(f"HEADS_{why.split()[0].upper()} no usable risk heads ({why}). Nothing was streamed.", flush=True)
            return NO_HEADS_EXIT
        if product:
            encoders.choose_into(os.environ, rest)
        rest = resolve_env_outputs(rest, os.environ)
        if os.environ.get(engine.STOP_ENV) == "1":
            engine.stop_on_stdin_close()
        from streamsafe.cli import main as m
    else:
        print((__doc__ or "").replace("<name>", CLI_EXE))
        return 0 if cmd in ("", "-h", "--help") else USAGE_EXIT
    return m(rest)


if __name__ == "__main__":
    sys.exit(main())


def translate_selftest() -> int:
    try:
        import ctranslate2  # noqa: F401
        import sentencepiece  # noqa: F401

        from streamsafe.translate.arabic import shape, visual_order

        print("TRANSLATE_LIBS_OK", visual_order(shape("اضغط X لإنهاء الرقص")) != "", flush=True)
        return 0
    except Exception as exc:  # report, the caller decides
        print("TRANSLATE_LIBS_FAILED", type(exc).__name__, exc, flush=True)
        return 1
