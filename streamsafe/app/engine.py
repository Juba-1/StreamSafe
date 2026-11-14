"""Starting, watching and stopping the engine process from the window.

The engine is the same program the command line runs (``<exe> run ...``), as a child process of the window:
  * it gets the product settings through its environment (``engine_env``): the ONNX backend, the model folder,
    the heads, the OBS state file, and the NVIDIA runtime with TensorRT only when that was checked to work;
  * the stream key never appears on a command line of ours: the full output address goes into the child's
    environment and the argument is only the name of that variable (``env:STREAMSAFE_OUTPUT_URL``);
  * it is stopped by closing its standard input (``stop_on_stdin_close`` in the child): the engine then ends
    like after Ctrl+C, closes its outputs and writes the session report. If the window dies, the pipe closes by
    itself and the engine stops too: no engine is left running without its window;
  * every line it prints is passed on with the stream key and the output address blanked out (``Redactor``).

What viewers see is decided inside the engine. The window only shows it: the state file the engine writes for
the OBS guard (``io/obs_link.py``) says "alive", the frame counter and "the whole picture is covered".
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from streamsafe.app import heads, paths, runtime_select
from streamsafe.app.settings import INPUT_SRT, OUTPUT_RTMP, Settings
from streamsafe.branding import PRODUCT_NAME

OUTPUT_ENV = "STREAMSAFE_OUTPUT_URL"
STOP_ENV = "STREAMSAFE_STOP_ON_STDIN"
PRODUCT_ENV = "STREAMSAFE_PRODUCT"
STALE_S = 1.5  # the frame counter not moving for this long while the process lives: shown as "not answering"
STOP_WAIT_S = 20.0

STOPPED, STARTING, PROTECTED, COVERED, STALLED, FAILED = "stopped", "starting", "protected", "covered", "stalled", "failed"


def exe_command(args: list[str]) -> list[str]:
    """The command line of the engine / tools: the console exe of the installed program, or this Python."""
    if paths.is_frozen():
        return [str(paths.app_dir() / (f"{PRODUCT_NAME}.exe" if sys.platform == "win32" else PRODUCT_NAME)), *args]
    return [sys.executable, "-m", "streamsafe.app.entry", *args]


def output_url(rtmp_url: str, key: str) -> str:
    return rtmp_url.strip().rstrip("/") + "/" + key


def run_args(settings: Settings, reports: Path) -> list[str]:
    """Arguments of ``run`` for these settings. Never holds the stream key."""
    source = f"srt://127.0.0.1:{settings.srt_port}?mode=listener" if settings.input == INPUT_SRT else f"device:{settings.camera.strip()}"
    sink = f"env:{OUTPUT_ENV}" if settings.output == OUTPUT_RTMP else "window"
    return ["run", "--input", source, "--output", sink, "--delay", f"{settings.delay:g}", "--report-dir", str(reports)]


def engine_env(
    base: Mapping[str, str],
    nvidia: bool,
    url: str = "",
    data_env: Mapping[str, str] | None = None,
    translate: bool = False,
    brand_image: str = "",
) -> dict[str, str]:
    """The environment of an engine / prepare / probe process. ``nvidia``: the NVIDIA runtime with TensorRT
    (engines are never built at "go live": ``TRT_BUILD=0``; a model without a ready engine runs on CUDA and the
    report says so). Otherwise the bundled DirectML runtime."""
    where = dict(base) if data_env is None else dict(data_env)
    env = dict(base)
    env.update(
        {
            PRODUCT_ENV: "1",
            "BACKEND": "onnx",
            "ONNX_DIR": str(paths.models_dir(where)),
            "HEAD_DIR": str(heads.best(where) or paths.bundled_heads_dir()),
            "OBS_STATE": str(paths.obs_state_file(where)),
            "TRT_CACHE": str(paths.trt_cache_dir(where)),
            "PYTHONUTF8": "1",
        }
    )
    if nvidia:
        env.update({runtime_select.RUNTIME_ENV: runtime_select.NVIDIA, "TRT": "1", "TRT_BUILD": "0"})
    else:
        env.update({runtime_select.RUNTIME_ENV: runtime_select.DIRECTML, "TRT": "0"})
    env.pop(OUTPUT_ENV, None)
    env.pop("TRANSLATE_AR", None)
    env.pop("TRANSLATE_DIR", None)
    if translate:  # the engine itself checks the pack and carries on without translation when it is not usable
        env.update({"TRANSLATE_AR": "1", "TRANSLATE_DIR": str(paths.translate_dir(where))})
    for name in ("SCENE_IMAGE", "SLATE_IMAGE", "SCENE_TEXT", "SLATE_TEXT"):
        env.pop(name, None)
    if brand_image and Path(brand_image).is_file():  # the streamer's logo alone on the full-screen cover
        env.update({"SCENE_IMAGE": brand_image, "SLATE_IMAGE": brand_image, "SCENE_TEXT": "", "SLATE_TEXT": ""})
    if url:
        env[OUTPUT_ENV] = url
    return env


class Redactor:
    """Blanks secrets out of text before it is shown or written anywhere."""

    MASK = "***"

    def __init__(self, *secrets: str) -> None:
        self._secrets = sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True)

    def clean(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, self.MASK)
        return text


@dataclass(frozen=True)
class State:
    alive: bool
    seq: int
    muted: bool
    full: bool


def read_state(path: Path) -> State | None:
    """The engine's state file (one line ``<alive> <seq> <muted> <full>``), None when missing or unreadable."""
    try:
        a, seq, muted, full = path.read_text(encoding="ascii").split()
        return State(a == "1", int(seq), muted == "1", full == "1")
    except (OSError, ValueError):
        return None


class StatusTracker:
    """Turns "is the process running" + the state file into the one word the window shows.

    PROTECTED is shown only on positive evidence: the engine says it is alive, its frame counter moved within
    ``STALE_S`` and the frame on screen is not the full cover. Everything else while the process runs is COVERED
    (the engine shows its slate while it loads and whenever it fails closed) or STALLED."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._seq: int | None = None
        self._moved = 0.0
        self._started = False

    def reset(self) -> None:
        self._seq, self._moved, self._started = None, 0.0, False

    def update(self, running: bool, state: State | None, failed: bool = False) -> str:
        if not running:
            self.reset()
            return FAILED if failed else STOPPED
        now = self._clock()
        if state is None or not state.alive:
            return STALLED if self._started and now - self._moved > STALE_S else STARTING
        if self._seq is None:  # first sight: may be a line left by an earlier run; wait until the counter moves
            self._seq, self._moved = state.seq, now
            return STARTING
        if state.seq != self._seq:
            self._seq, self._moved, self._started = state.seq, now, True
        if not self._started:
            return STALLED if now - self._moved > 10 * STALE_S else STARTING
        if now - self._moved > STALE_S:
            return STALLED
        return COVERED if state.full else PROTECTED


def stop_on_stdin_close(stream=None, interrupt: Callable[[], None] | None = None) -> threading.Thread:
    """In the engine process: end the run cleanly when standard input closes or says "stop" (the window's Stop
    button, or the window itself gone). The main thread gets a KeyboardInterrupt, which the engine handles like
    Ctrl+C: outputs closed, report written."""
    import _thread

    stream = sys.stdin if stream is None else stream
    interrupt = interrupt or _thread.interrupt_main

    def watch() -> None:
        try:
            for line in stream:
                if line.strip().lower() == "stop":
                    break
        except (OSError, ValueError):
            pass
        interrupt()

    thread = threading.Thread(target=watch, name="stop-watch", daemon=True)
    thread.start()
    return thread


def _no_window_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)  # a console child of a window app: no console flashes up


class EngineProcess:
    """One child process whose output lines are handed to ``on_line`` (already redacted)."""

    def __init__(self, on_line: Callable[[str], None], redactor: Redactor | None = None) -> None:
        self._on_line = on_line
        self._redactor = redactor or Redactor()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._log = None
        self.stop_requested = False

    def redactor_secrets(self, *secrets: str) -> None:
        """The secrets (stream key, full output address) to blank out of every line from now on."""
        self._redactor = Redactor(*secrets)

    def start(self, cmd: list[str], env: Mapping[str, str], log_path: Path | None = None) -> None:
        """``log_path``: every (redacted) output line is also kept there, so a tester can send it when something
        goes wrong (the window keeps only the last lines)."""
        if self.running:
            raise RuntimeError("already running")
        self.stop_requested = False
        self._log = None
        if log_path is not None:
            with contextlib.suppress(OSError):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed when the output ends
        self._proc = subprocess.Popen(
            cmd,
            env=dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=_no_window_flags(),
        )
        self._reader = threading.Thread(target=self._read, args=(self._proc,), name="engine-output", daemon=True)
        self._reader.start()

    def _read(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        log = self._log
        try:
            with contextlib.suppress(OSError, ValueError):
                for line in proc.stdout:
                    clean = self._redactor.clean(line.rstrip("\r\n"))
                    if log is not None:
                        with contextlib.suppress(OSError, ValueError):
                            log.write(f"{time.strftime('%H:%M:%S')} {clean}\n")
                            log.flush()
                    self._on_line(clean)
        finally:
            if log is not None:
                with contextlib.suppress(OSError, ValueError):
                    log.write(f"{time.strftime('%H:%M:%S')} [engine ended, exit {proc.poll()}]\n")
                    log.close()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def exit_code(self) -> int | None:
        return None if self._proc is None else self._proc.poll()

    def stop(self, wait_s: float = STOP_WAIT_S) -> int | None:
        """Ask the engine to end (clean stop, report written); kill it if it has not ended after ``wait_s``."""
        proc = self._proc
        if proc is None:
            return None
        self.stop_requested = True
        if proc.poll() is None:
            with contextlib.suppress(OSError, ValueError):
                assert proc.stdin is not None
                proc.stdin.close()
            try:
                proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                kill_tree(proc)
        if self._reader is not None:
            self._reader.join(timeout=5)
        return proc.poll()


def kill_tree(proc: subprocess.Popen) -> None:
    """Last resort for an engine that does not end: it and the processes it started (analyzer, FFmpeg)."""
    if sys.platform == "win32":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
                check=False,
                creationflags=_no_window_flags(),
            )
    with contextlib.suppress(OSError):
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def run_tool(
    args: list[str],
    env: Mapping[str, str],
    on_line: Callable[[str], None],
    cancel: threading.Event | None = None,
    timeout_s: float = 3600.0,
) -> int:
    """Run ``<exe> <args>`` to its end (prepare, gpu-probe), handing over its lines. Returns the exit code;
    -1 when it was cancelled or ran out of time (then it is killed)."""
    proc = subprocess.Popen(
        exe_command(args),
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=_no_window_flags(),
    )
    assert proc.stdout is not None

    def read() -> None:
        with contextlib.suppress(OSError, ValueError):
            for line in proc.stdout:  # type: ignore[union-attr]
                on_line(line.rstrip("\r\n"))

    reader = threading.Thread(target=read, name="tool-output", daemon=True)
    reader.start()
    end = time.monotonic() + timeout_s
    while proc.poll() is None:
        if (cancel is not None and cancel.is_set()) or time.monotonic() > end:
            kill_tree(proc)
            reader.join(timeout=5)
            return -1
        time.sleep(0.1)
    reader.join(timeout=5)
    return proc.returncode


def last_report(reports: Path) -> Path | None:
    """The newest tester session report (``session_*.txt``), None when there is none yet."""
    try:
        found = [p for p in reports.glob("session_*.txt") if p.is_file()]
        return max(found, key=lambda p: p.stat().st_mtime) if found else None
    except OSError:
        return None


def open_file(path: Path) -> None:
    """Open a file or folder with the program Windows has for it (the report: Notepad)."""
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
