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

from streamsafe.app import heads, paths, plan, runtime_select
from streamsafe.app.settings import INPUT_FILE, INPUT_SRT, OUTPUT_RECORD, OUTPUT_RTMP, RESOLUTIONS, Settings
from streamsafe.branding import PRODUCT_NAME

OUTPUT_ENV = "STREAMSAFE_OUTPUT_URL"
OUTPUT2_ENV = "STREAMSAFE_OUTPUT_URL_2"  # the extra platform (multistream through StreamSafe)
STOP_ENV = "STREAMSAFE_STOP_ON_STDIN"
PRODUCT_ENV = "STREAMSAFE_PRODUCT"
STALE_S = 1.5  # the frame counter not moving for this long while the process lives: shown as "not answering"
STOP_WAIT_S = 40.0  # a clean stop: analyzer grace + outputs closed + report written

STOPPED, STARTING, PROTECTED, COVERED, STALLED, FAILED = "stopped", "starting", "protected", "covered", "stalled", "failed"


def exe_command(args: list[str]) -> list[str]:
    """The command line of the engine / tools: the console exe of the installed program, or this Python."""
    if paths.is_frozen():
        return [str(paths.app_dir() / (f"{PRODUCT_NAME}.exe" if sys.platform == "win32" else PRODUCT_NAME)), *args]
    return [sys.executable, "-m", "streamsafe.app.entry", *args]


def output_url(rtmp_url: str, key: str, platform: str = "custom") -> str:
    """The full output address (it holds the key). Normalised by ``ingest.resolve``: a Kick / IVS server pasted
    without ``/app`` gets it, spaces are removed; "" when the address or the key cannot be used."""
    from streamsafe.app import ingest

    target = ingest.resolve(rtmp_url, key, platform)
    return "" if target.problems else target.url


AUTO_DELAY_FAST, AUTO_DELAY_SLOW = 2.0, 3.0


def delay_for(settings: Settings, fast: bool) -> float:
    """The stream delay. Automatic (0): 2 s on the fast NVIDIA mode, 3 s otherwise - the delay is not what makes
    the analysis keep up (the decision rate does), but the frames waiting in it are what the look-ahead covers
    when a person turns out risky: a slower PC decides a little later, one more second keeps those frames covered."""
    return settings.delay if settings.delay > 0 else (AUTO_DELAY_FAST if fast else AUTO_DELAY_SLOW)


def follow_obs(settings: Settings, video: tuple[int, int, float] | None) -> None:
    """Live from OBS: our picture is OBS's own (the smallest of our sizes that holds it, 30 or 60 fps), so the
    streamer's quality is kept without setting anything twice."""
    if not settings.follow_obs or settings.input != INPUT_SRT or video is None:
        return
    w, h, _fps = video
    short = min(w, h)
    fits = [name for name, (_rw, rh) in sorted(RESOLUTIONS.items(), key=lambda kv: kv[1][1]) if rh >= short]
    settings.resolution = fits[0] if fits else "2160p"
    settings.fps = 30  # 30 for every live stream: 60 doubled the work and the upload and broke up on mid-range PCs
    # OBS's exact shape is kept: a vertical (TikTok) or any other canvas is never squeezed into 16:9
    settings.obs_size = f"{w - w % 2}x{h - h % 2}" if w >= 16 and h >= 16 else ""


def live_size(settings: Settings) -> tuple[int, int]:
    """The live picture's size: OBS's own shape when following OBS, else the chosen 16:9 size. Platforms other
    than YouTube take at most 1080 on the short side: more is scaled down here, keeping the shape."""
    w, h = RESOLUTIONS.get(settings.resolution, RESOLUTIONS["1080p"])
    if settings.follow_obs and settings.input == INPUT_SRT and settings.obs_size:
        with contextlib.suppress(ValueError):
            w, h = (int(v) for v in settings.obs_size.lower().split("x"))
    if settings.output == OUTPUT_RTMP and settings.platform not in ("youtube", "custom") and min(w, h) > 1080:
        k = 1080 / min(w, h)
        w, h = round(w * k / 2) * 2, round(h * k / 2) * 2
    return w, h


def run_args(settings: Settings, reports: Path, record_file: str = "", fast: bool = True) -> list[str]:
    """Arguments of ``run`` for these settings. Never holds the stream key.

    Quality: a live picture runs at the chosen resolution and frame rate (``settings.resolution`` / ``fps``, the
    same as OBS sends); a recorded video keeps its own size and rate. The stream's bitrate is the platform's own
    figure for that size (``ingest.bitrate_for``); a recording's is set by the engine for its size."""
    if settings.input == INPUT_SRT:
        source = f"srt://127.0.0.1:{settings.srt_port}?mode=listener"
    elif settings.input == INPUT_FILE:
        source = f"file:{settings.input_file}"
    else:
        sound = settings.camera_sound.strip()
        source = f"device:{settings.camera.strip()}" + (f"|{sound}" if sound else "")
    sink = (
        f"env:{OUTPUT_ENV}" if settings.output == OUTPUT_RTMP else f"record:{record_file}" if settings.output == OUTPUT_RECORD else "window"
    )
    args = ["run", "--input", source, "--output", sink]
    if settings.output == OUTPUT_RTMP and settings.platform2:
        args += ["--extra-output", f"env:{OUTPUT2_ENV}"]
    args += ["--delay", f"{delay_for(settings, fast):g}", "--report-dir", str(reports)]
    if settings.input == INPUT_FILE:
        args += ["--fps", "auto"]
    else:
        w, h = live_size(settings)
        args += ["--size", f"{w}x{h}", "--fps", str(settings.fps)]
        if settings.output == OUTPUT_RTMP:
            from streamsafe.app import ingest

            kbps = ingest.bitrate_for(settings.platform, min(w, h), settings.fps)
            if settings.follow_obs and settings.input == INPUT_SRT and settings.obs_kbps >= 1000:
                kbps = min(kbps, settings.obs_kbps)  # never more than the streamer's own OBS sends: their upload
            args += ["--bitrate", str(kbps)]
    return args


@dataclass
class LinkState:
    """What the engine last said about its two links (``io/link.py``), for the window."""

    input: str = ""  # "" / waiting / connected / disconnected / failed
    sent: tuple = ()  # (width, height, fps, kbps) of what OBS sends, when FFmpeg said it
    platform: str = ""  # "" / connecting / connected / lost / failed
    platform2: str = ""  # the extra platform: the same words
    port: int = 0
    reason: str = ""  # the last platform / input problem, a word of io/link.py
    retry_s: float = 0.0
    record_file: str = ""
    recorded: bool = False
    stopped: str = ""  # why the run ended ("" while it runs)
    detail: str = ""

    def take(self, event: dict) -> None:
        name = event.get("event", "")
        if event.get("out"):  # the extra platform: its own line, never the main platform's state
            self.platform2 = name.replace("platform_", "")
            return
        if name == "waiting_obs":
            self.input = "waiting" if self.input in ("", "waiting") else "disconnected"
            self.port = int(event.get("port") or 0)
        elif name == "obs_connected":
            self.input = "connected"
        elif name == "obs_format":  # what OBS sends: the window warns when the protection runs smaller / slower
            self.sent = (
                int(event.get("width") or 0),
                int(event.get("height") or 0),
                float(event.get("fps") or 0),
                int(event.get("kbps") or 0),
            )
        elif name == "obs_disconnected":
            self.input = "disconnected"
        elif name == "input_failed":
            self.input, self.reason = "failed", str(event.get("reason", ""))
        elif name == "platform_connecting":
            self.platform = "connecting" if self.platform != "lost" else "lost"
        elif name == "platform_connected":
            self.platform, self.reason = "connected", ""
        elif name == "platform_lost":
            self.platform, self.reason = "lost", str(event.get("reason", ""))
            self.retry_s = float(event.get("retry_s") or 0)
        elif name == "platform_failed":
            self.platform, self.reason = "failed", str(event.get("reason", ""))
        elif name == "recording":
            self.record_file, self.recorded = str(event.get("file", "")), False
        elif name == "recorded":
            self.record_file, self.recorded = str(event.get("file", "")), True
        elif name == "stopped":
            self.stopped, self.detail = str(event.get("reason", "")), str(event.get("detail", ""))


def engine_env(
    base: Mapping[str, str],
    nvidia: bool,
    url: str = "",
    data_env: Mapping[str, str] | None = None,
    translate: bool = False,
    brand_image: str = "",
    url2: str = "",
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
    env.pop(OUTPUT2_ENV, None)
    env.pop("TRANSLATE_AR", None)
    env.pop("TRANSLATE_DIR", None)
    if translate:  # the engine itself checks the pack and carries on without translation when it is not usable
        env.update({"TRANSLATE_AR": "1", "TRANSLATE_DIR": str(paths.translate_dir(where))})
    for name in ("SCENE_IMAGE", "SLATE_IMAGE", "SCENE_TEXT", "SLATE_TEXT", "LOGO"):
        env.pop(name, None)
    # the covers carry OUR logo (the clothing's chest and the full-screen card). A Pro subscription may put the
    # streamer's / sponsor's logo there instead (plan.py); everyone else sees StreamSafe's.
    ours_card, ours_chest = paths.bundled_brand_dir() / "logo.png", paths.bundled_brand_dir() / "logo_rgba.png"
    own = brand_image if brand_image and Path(brand_image).is_file() and plan.allows("custom_brand", where) else ""
    card = own or (str(ours_card) if ours_card.is_file() else "")
    chest = own or (str(ours_chest) if ours_chest.is_file() else "")
    if card:
        env.update({"SCENE_IMAGE": card, "SLATE_IMAGE": card, "SCENE_TEXT": "", "SLATE_TEXT": ""})
    if chest:
        env["LOGO"] = chest
    if url:
        env[OUTPUT_ENV] = url
    if url2:
        env[OUTPUT2_ENV] = url2
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


from streamsafe.io.sinks import priority_flags  # noqa: E402


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
            # above normal (as OBS does for its own streaming): the frame pacing never waits behind the game.
            # Its analyzer child starts at normal priority (Windows' rule), so the game keeps its share.
            creationflags=_no_window_flags() | priority_flags(),
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
