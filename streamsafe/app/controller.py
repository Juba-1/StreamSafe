"""What the window does, without any window: the settings and the stream key, the system check, installing the
models, the NVIDIA fast mode, the update check, and starting / watching / stopping the engine.

``ui.py`` only draws this and calls it. Everything here is testable without a screen, and every failure ends on the
safe side: no start when anything is unclear, and a running engine that stops or stalls is shown as covered.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from streamsafe import session_report
from streamsafe.app import engine, fetch, heads, ingest, keystore, modelpack, nvidia_pack, paths, release, updates
from streamsafe.app import license as lic
from streamsafe.app import settings as settings_mod
from streamsafe.app.settings import INPUT_SRT, OUTPUT_RECORD, OUTPUT_RTMP, OUTPUT_WINDOW, THR_SCALE, Settings
from streamsafe.branding import PRODUCT_NAME
from streamsafe.io import link
from streamsafe.translate import pack as translate_pack

Log = Callable[[str], None]
Progress = Callable[[str, int, int], None]  # (what, done, of); of 0 = unknown


@dataclass
class CheckResult:
    verdict: str = ""  # "ok" / "warn" / "fail" / "" (not run, or could not run)
    checks: list[dict] = field(default_factory=list)
    error: str = ""

    def item(self, name: str) -> dict | None:
        return next((c for c in self.checks if c.get("name") == name), None)


def parse_check_output(lines: list[str]) -> CheckResult:
    """The JSON object printed by ``check --json`` (it may be preceded by other lines)."""
    text = "\n".join(lines)
    start = text.find("{")
    try:
        data = json.loads(text[start:]) if start >= 0 else {}
        return CheckResult(str(data["verdict"]), list(data["checks"]))
    except (ValueError, KeyError, TypeError):
        return CheckResult(error=" ".join(lines[-3:])[:300] or "no answer")


class Controller:
    def __init__(
        self,
        env: MutableMapping[str, str] | None = None,
        protector: keystore.Protector | None = None,
        log: Log | None = None,
        allow_http: bool = False,
    ) -> None:
        self.env = os.environ if env is None else env
        self.log = log or (lambda _line: None)
        self.allow_http = allow_http
        self.key2 = ""  # the extra platform's stream key, as typed (the window sets it before a start)
        self.preview = False  # the running engine shows a local window only (``start(preview=True)``)
        self.settings: Settings = settings_mod.load(paths.settings_file(self.env))
        guard = keystore.default_protector() if protector is None else protector
        self.keys = keystore.KeyStore(paths.key_file(self.env), guard)
        kf = paths.key_file(self.env)
        self.keys2 = keystore.KeyStore(kf.with_name(kf.stem + "_2" + kf.suffix), guard)  # the extra platform's key
        self.engine = engine.EngineProcess(self._engine_line, engine.Redactor())
        self.tracker = engine.StatusTracker()
        self.check_result = CheckResult()
        self.license_status = lic.Status(True) if not release.LICENSE_REQUIRED else lic.Status(False, "no_license")
        self._license_at = 0.0  # monotonic time of the last licence check
        self.license_stopped = False  # protection was stopped because the subscription is no longer valid
        self.use_nvidia = False  # decided by the last check: engines ready AND the NVIDIA libraries in place
        self.last_exit: int | None = None
        self.update: updates.Update | None = None
        self._engine_lines: list[str] = []
        self.link = engine.LinkState()
        self.port_moved_from = 0  # the port that was busy when the last start picked another one (0: it was free)
        self._report_line = ""  # the engine's own report of this run
        self._closed_out = True  # the end of the last run has been written up

    # --- settings and key ---------------------------------------------------------------------------------
    def save_settings(self) -> None:
        try:
            settings_mod.save(paths.settings_file(self.env), self.settings)
        except OSError as exc:
            self.log(f"settings not saved: {exc}")

    def stored_key(self) -> str:
        return self.keys.load() if self.settings.remember_key else ""

    def stored_key2(self) -> str:
        return self.keys2.load() if self.settings.remember_key else ""

    def remember2(self, key: str) -> None:
        with contextlib.suppress(keystore.NoSecureStoreError, OSError, ValueError):
            if self.settings.remember_key and key and self.settings.platform2:
                self.keys2.save(key)
            else:
                self.keys2.clear()

    def remember(self, key: str) -> bool:
        """Store or forget the key as the settings say. False when the PC has no protected store."""
        if not (self.settings.remember_key and key):
            self.keys.clear()
            return True
        try:
            self.keys.save(key)
        except (keystore.NoSecureStoreError, OSError, ValueError):
            self.settings.remember_key = False
            return False
        return True

    # --- state of the installation ------------------------------------------------------------------------
    def models_problem(self) -> str:
        return modelpack.problem(paths.models_dir(self.env))

    def heads_problem(self) -> str:
        return heads.why_not(self.env)

    def days_left(self) -> int | None:
        return updates.days_left()

    def expired(self) -> bool:
        return updates.expired()

    def blockers(self, key: str = "", settings: Settings | None = None) -> list[str]:
        """i18n keys (``why_*`` / ``bad_*``) of what stops a start. Empty: it may start."""
        s = settings or self.settings
        out: list[str] = []
        if self.engine.running:
            return ["why_running"]
        if self.expired():
            out.append("why_expired")
        out += s.problems()
        if s.output == OUTPUT_RTMP and s.platform2 and "bad_key" in self.target2(self.key2).problems:
            out.append("bad_key2")
        if s.output == OUTPUT_RTMP and "bad_key" in self.target(key).problems:
            out.append("bad_key")
        if self.models_problem():
            out.append("why_models")
        why = self.heads_problem()
        if why:
            out.append("why_heads_placeholder" if why == "placeholder" else "why_heads")
        if self.check_result.verdict == "fail":
            out.append("why_check")
        if release.LICENSE_REQUIRED and not self.license_status.ok:
            out.append("why_license_" + (self.license_status.reason or "invalid"))
        return out

    # --- the subscription --------------------------------------------------------------------------------------
    def check_license(self, post: lic.Post | None = None) -> lic.Status:
        """Ask Dodo (network: call it off the window's thread). Without LICENSE_REQUIRED every copy may run."""
        if not release.LICENSE_REQUIRED:
            self.license_status = lic.Status(True, plan=lic.load(self.env).plan if lic.load(self.env) else "basic")
        else:
            self.license_status = lic.check(self.env, post or lic.http_post)
        self._license_at = time.monotonic()
        return self.license_status

    def activate_license(self, key: str, post: lic.Post | None = None) -> lic.Status:
        self.license_status = lic.activate(key, self.env, post or lic.http_post)
        self._license_at = time.monotonic()
        return self.license_status

    def deactivate_license(self, post: lic.Post | None = None) -> bool:
        done = lic.deactivate(self.env, post or lic.http_post)
        self.license_status = lic.Status(not release.LICENSE_REQUIRED, "no_license")
        return done

    def _recheck_license_while_running(self) -> None:
        """Every few hours while protection runs: a subscription that ended stops it (the viewers get the normal
        end of stream, never an unprotected picture)."""
        if not release.LICENSE_REQUIRED or time.monotonic() - self._license_at < release.LICENSE_RECHECK_HOURS * 3600:
            return
        self._license_at = time.monotonic()

        def work() -> None:
            if not self.check_license().ok and self.engine.running:
                self.license_stopped = True
                self.stop()

        threading.Thread(target=work, name="license-check", daemon=True).start()

    # --- the system check -----------------------------------------------------------------------------------
    def tool_env(self, nvidia: bool | None = None) -> dict[str, str]:
        wants = nvidia_pack.ready(paths.nvidia_dir(self.env)) if nvidia is None else nvidia
        return engine.engine_env(os.environ, wants, data_env=self.env)

    def run_check(self, cancel: threading.Event | None = None) -> CheckResult:
        lines: list[str] = []
        code = engine.run_tool(["check", "--json", "--quick"], self.tool_env(), lines.append, cancel, timeout_s=300)
        result = parse_check_output(lines) if code >= 0 else CheckResult(error="cancelled")
        self.check_result = result
        eng = result.item("engines")
        self.use_nvidia = bool(eng and eng.get("status") == "ok" and nvidia_pack.ready(paths.nvidia_dir(self.env)))
        return result

    def has_nvidia_gpu(self) -> bool:
        gpu = self.check_result.item("gpu")
        return bool(gpu) and any(w in str(gpu.get("detail", "")).lower() for w in ("nvidia", "geforce", "quadro", "rtx"))

    # --- models -------------------------------------------------------------------------------------------
    def install_models(
        self, source: str, location: str = "", progress: Progress | None = None, cancel: threading.Event | None = None
    ) -> None:
        """``source``: "download" (``release.MODEL_BASE_URL``), "zip" or "folder" (``location``).
        Raises modelpack.SourceError / PackError."""
        models = paths.models_dir(self.env)
        if source == "download":
            modelpack.download_from_base(release.MODEL_BASE_URL, models, paths.downloads_dir(self.env), progress, cancel, self.allow_http)
        elif source == "zip":
            modelpack.install_zip(Path(location), models, "", progress, cancel)
        elif source == "folder":
            modelpack.install_folder(Path(location), models, progress, cancel)
        else:
            raise ValueError(source)

    # --- Arabic translation of on-screen text ---------------------------------------------------------------
    BRAND_MAX = 1280  # px: a bigger logo is shrunk when it is chosen

    def set_brand_image(self, source: str) -> str:
        """Use ``source`` as the logo on the full-screen cover: checked, shrunk, and kept as a PNG copy in the data
        folder (the original may move). '' when done, else the i18n key of the problem. An empty ``source`` removes it."""
        if not source:
            self.settings.brand_image = ""
            self.save_settings()
            return ""
        import cv2
        import numpy as np

        try:  # np.fromfile + imdecode: cv2.imread cannot open non-ASCII paths on Windows
            img = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_UNCHANGED)
        except (OSError, ValueError):
            img = None
        if img is None or img.ndim < 2 or min(img.shape[:2]) < 16:
            return "brand_bad"
        scale = self.BRAND_MAX / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        folder = paths.brand_dir(self.env)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            ok, data = cv2.imencode(".png", img)
            if not ok:
                return "brand_bad"
            target = folder / "logo.png"
            target.write_bytes(data.tobytes())
        except OSError:
            return "brand_bad"
        self.settings.brand_image = str(target)
        self.save_settings()
        return ""

    def translate_problem(self) -> str:
        """ "" when the translation files are installed and are the pinned ones (sizes; the engine checks every hash)."""
        return translate_pack.problem(paths.translate_dir(self.env))

    def install_translate(self, progress: Progress | None = None, cancel: threading.Event | None = None) -> str:
        """Download the translation pack (``release.TRANSLATE_FILES``). Returns "" on success, otherwise why not.
        The stream never depends on it: a failure only leaves translation off."""
        try:
            translate_pack.install_download(
                release.TRANSLATE_BASE_URL,
                paths.translate_dir(self.env),
                paths.downloads_dir(self.env),
                progress=progress,
                cancel=cancel,
                allow_http=self.allow_http,
            )
        except (fetch.DownloadError, translate_pack.PackError, OSError) as exc:
            return str(exc)
        return ""

    # --- NVIDIA fast mode ---------------------------------------------------------------------------------
    def enable_nvidia(self, progress: Progress | None = None, cancel: threading.Event | None = None) -> str:
        """Download NVIDIA's libraries, check that ONNX Runtime reaches the card with them, build the engines.
        Returns "" on success, otherwise why not (the app stays in the normal mode)."""
        try:
            nvidia_pack.install(paths.nvidia_dir(self.env), paths.downloads_dir(self.env), progress, cancel, allow_http=self.allow_http)
        except nvidia_pack.NvidiaPackError as exc:
            return str(exc)
        env = self.tool_env(True)
        probe: list[str] = []
        engine.run_tool(["gpu-probe"], env, probe.append, cancel, timeout_s=300)
        if not any('"device": "cuda"' in line for line in probe):
            return "ONNX Runtime cannot reach the NVIDIA card: " + " ".join(probe[-1:])[:200]

        def line(text: str) -> None:
            try:
                ev = json.loads(text)
                if progress and ev.get("of"):
                    progress(f"{ev.get('name', '')}", int(ev.get("step", 0)), int(ev["of"]))
            except (ValueError, TypeError, AttributeError):
                self.log(text)

        code = engine.run_tool(["prepare", "--json"], env, line, cancel, timeout_s=3600)
        return "" if code == 0 else "the engines could not be built"

    # --- updates ------------------------------------------------------------------------------------------
    def check_updates(self) -> updates.Update | None:
        """None when updates are off or there is nothing new; UpdateError is turned into None (carry on)."""
        try:
            found = updates.check(self.settings.update_serial)
        except updates.UpdateError as exc:
            self.log(f"update check: {exc}")
            return None
        self.update = found
        return found

    def update_wanted(self) -> tuple[bool, bool]:
        """(a new app version exists, new model / head data exists)."""
        u = self.update
        if u is None:
            return False, False
        data = updates.models_differ(u, modelpack.current_shas(paths.models_dir(self.env))) or updates.heads_newer(
            u, heads.version(heads.best(self.env) or paths.bundled_heads_dir())
        )
        return u.app_is_newer(), data

    def apply_update(self, progress: modelpack.Progress | None = None, cancel: threading.Event | None = None) -> list[str]:
        if self.update is None:
            return []
        done = updates.apply_data(
            self.update,
            models=paths.models_dir(self.env),
            heads_folder=paths.heads_dir(self.env),
            downloads=paths.downloads_dir(self.env),
            installed_heads_version=heads.version(heads.best(self.env) or paths.bundled_heads_dir()),
            progress=progress,
            cancel=cancel,
            allow_http=self.allow_http,
        )
        self.settings.update_serial = max(self.settings.update_serial, self.update.serial)
        self.save_settings()
        return done

    def download_app_update(self, progress: modelpack.Progress | None = None, cancel: threading.Event | None = None) -> Path:
        """The new installer, downloaded and checked against the signed manifest's sha256 (never run unchecked)."""
        u = self.update
        if u is None or not u.app_sha256 or not u.app_url:
            raise updates.UpdateError("no installer in the update")
        dest = paths.downloads_dir(self.env) / f"{PRODUCT_NAME}-Setup-{u.app_version}.exe"
        return fetch.download(u.app_url, dest, u.app_sha256, u.app_bytes, progress, cancel, self.allow_http)

    @staticmethod
    def run_installer(installer: Path) -> None:
        """Installs silently over this version and opens the app again (the window closes itself right after)."""
        import subprocess

        flags = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/RELAUNCH"]
        subprocess.Popen([str(installer), *flags], close_fds=True)

    def can_roll_back(self) -> bool:
        return modelpack.previous(paths.models_dir(self.env)).is_dir()

    def roll_back(self) -> None:
        modelpack.rollback(paths.models_dir(self.env))

    # --- the two links: OBS -> us, us -> the platform ------------------------------------------------------
    def target(self, key: str) -> ingest.Target:
        """Server and key as they will be used (``ingest.resolve``). Holds the key: never log it."""
        return ingest.resolve(self.settings.rtmp_url, key, self.settings.platform)

    def target2(self, key: str) -> ingest.Target:
        """The extra platform's server and key. Holds the key: never log it."""
        return ingest.resolve(self.settings.rtmp_url2, key, self.settings.platform2)

    def take_pasted(self, server: str, key: str) -> tuple[str, str]:
        """What the two boxes should show after a paste: a whole address is split into (server, key), spaces go,
        a Kick server gets its ``/app``. Values that cannot be used come back as typed (``blockers`` says why)."""
        t = ingest.resolve(server, key, self.settings.platform)
        if "bad_rtmp" in t.problems:
            return ingest.squeeze(server), ingest.squeeze(key)
        return t.server, t.key

    def obs_server(self) -> str:
        """What goes into OBS: Settings -> Stream -> Service "Custom" -> Server."""
        return f"srt://127.0.0.1:{self.settings.srt_port}"

    def free_port(self) -> bool:
        """Before a start: when the SRT port is taken by another program, move to a free one (kept in the
        settings). True when it moved - OBS then needs the new server address (``obs_server``)."""
        s = self.settings
        self.port_moved_from = 0
        if s.input != INPUT_SRT or not 1024 <= s.srt_port <= 65535 or link.udp_port_free(s.srt_port):
            return False
        self.port_moved_from, s.srt_port = s.srt_port, link.free_udp_port(s.srt_port + 1)
        self.save_settings()
        return True

    def record_file(self) -> str:
        """A new file name for a recording, in the folder of the settings (default: Videos/<product>)."""
        folder = Path(self.settings.record_dir) if self.settings.record_dir else Path.home() / "Videos" / PRODUCT_NAME
        return str(folder / f"{PRODUCT_NAME}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    def test_connection(self, key: str, cancel: threading.Event | None = None) -> str:
        """Send a few seconds of the black slate to the platform. "" when it took the stream, else a reason word
        (``why_<reason>`` in i18n). The channel is live for those seconds: the window warns first."""
        t = self.target(key)
        if t.problems:
            return t.problems[0]
        env = engine.engine_env(os.environ, self.use_nvidia, t.url, self.env)
        clean = engine.Redactor(t.key, t.url)
        lines: list[str] = []

        def line(text: str) -> None:
            lines.append(clean.clean(text))
            self.log(lines[-1])

        code = engine.run_tool(["link-test", "--output", f"env:{engine.OUTPUT_ENV}"], env, line, cancel, timeout_s=90)
        if code == 0 and any(text.startswith("LINK_TEST ok") for text in lines):
            return ""
        failed = next((text for text in reversed(lines) if text.startswith("LINK_TEST failed")), "")
        return (failed.split()[2] if len(failed.split()) > 2 else "") or ("cancelled" if code < 0 else "unknown")

    def devices(self) -> dict[str, list[str]]:
        """The capture devices of this PC (cameras, capture cards / their sound inputs), for the settings."""
        lines: list[str] = []
        engine.run_tool(["devices"], self.tool_env(False), lines.append, timeout_s=40)
        for text in lines:
            if text.startswith("DEVICES "):
                with contextlib.suppress(ValueError):
                    found = json.loads(text[len("DEVICES ") :])
                    return {"video": list(found.get("video", [])), "audio": list(found.get("audio", []))}
        return {"video": [], "audio": []}

    # --- the engine -----------------------------------------------------------------------------------------
    def _engine_line(self, line: str) -> None:
        self._engine_lines = [*self._engine_lines[-60:], line]
        event = link.parse(line)
        if event is not None:
            self.link.take(event)
        elif line.startswith("SESSION_REPORT "):
            self._report_line = line[len("SESSION_REPORT ") :].strip()
        self.log(line)

    KEEP_LOGS = 10

    def _new_log(self) -> Path:
        """A fresh engine log in the logs folder of the data folder; only the last ``KEEP_LOGS`` are kept."""
        folder = paths.logs_dir(self.env)
        with contextlib.suppress(OSError):
            old = sorted(folder.glob("engine_*.log"))
            for f in old[: max(0, len(old) - self.KEEP_LOGS + 1)]:
                f.unlink()
        return folder / f"engine_{time.strftime('%Y%m%d_%H%M%S')}.log"

    def panic_file(self) -> Path:
        """Exists while the streamer's panic key covers the whole screen (the engine looks for it)."""
        return paths.data_dir(self.env) / "panic.flag"

    def set_panic(self, on: bool) -> None:
        f = self.panic_file()
        with contextlib.suppress(OSError):
            if on:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("1", encoding="ascii")
            else:
                f.unlink(missing_ok=True)

    @property
    def panic(self) -> bool:
        return self.panic_file().exists()

    def start(self, key: str, preview: bool = False) -> None:
        """Start the engine. Raises RuntimeError with the blocking i18n key when it must not start.
        ``preview``: the protected picture goes to a window on this PC only, nothing to the platform."""
        if release.LICENSE_REQUIRED:
            self.check_license()  # every start asks Dodo (a lapsed subscription does not start)
        s = replace(self.settings, output=OUTPUT_WINDOW) if preview else self.settings
        why = self.blockers(key, s)
        if why:
            raise RuntimeError(why[0])
        self.license_stopped = False
        self.preview = preview
        self.free_port()
        target = self.target(key) if s.output == OUTPUT_RTMP else ingest.Target("", "")
        url = target.url
        t2 = self.target2(self.key2) if s.output == OUTPUT_RTMP and s.platform2 else ingest.Target("", "")
        env = engine.engine_env(
            os.environ,
            self.use_nvidia,
            url,
            self.env,
            translate=s.translate_ar and not self.translate_problem(),
            brand_image=s.brand_image,
            url2=t2.url,
        )
        env["COVER_MODE"] = s.cover_mode
        env["THR_SCALE"] = str(THR_SCALE.get(s.sensitivity, 1.0))
        self.set_panic(False)  # a panic left on by an earlier run starts this one uncovered only if pressed again
        env["PANIC_FILE"] = str(self.panic_file())
        state = paths.obs_state_file(self.env)
        state.unlink(missing_ok=True)  # a line left by an earlier run must not look like this run's
        state.parent.mkdir(parents=True, exist_ok=True)
        self.engine.redactor_secrets(*(x for x in (key, target.key, url, self.key2, t2.key, t2.url) if x))
        self.tracker.reset()
        self.last_exit = None
        self._engine_lines = []
        self.link = engine.LinkState()
        self._report_line, self._closed_out = "", False
        record = self.record_file() if s.output == OUTPUT_RECORD else ""
        self.engine.start(
            engine.exe_command(engine.run_args(s, paths.reports_dir(self.env), record, fast=self.use_nvidia)), env, self._new_log()
        )

    def status(self) -> str:
        running = self.engine.running
        if running:
            self._recheck_license_while_running()
        if not running and self.engine.exit_code is not None and not self.engine.stop_requested:
            self.last_exit = self.engine.exit_code
        if not running and self.engine.exit_code is not None:
            self.close_out()
        failed = self.last_exit not in (None, 0)
        return self.tracker.update(running, engine.read_state(paths.obs_state_file(self.env)), failed)

    def close_out(self) -> Path | None:
        """Once per run, when the engine has ended: every start that did not end well leaves ONE report that says
        why, with the engine's last lines under it. The engine's own report gets the lines added; an engine that
        wrote none (refused at start, crashed, killed) gets a report written here."""
        if self._closed_out:
            return None
        self._closed_out = True
        code = self.engine.exit_code
        time.sleep(0.05)  # the reader thread may still be handing over the last lines
        lines = list(self._engine_lines)
        try:
            if self._report_line and Path(self._report_line).is_file():
                path = Path(self._report_line)
                if code != 0 or self.link.stopped in session_report.FAILED_REASONS:
                    session_report.add_log(path, lines)
                return path
            reason = self.link.stopped if self.link.stopped in session_report.FAILED_REASONS else "start_failed"
            detail = f"exit code {code}: {self.exit_reason()} {self.link.detail}".strip()
            return session_report.write_failure(reason, detail, lines, paths.reports_dir(self.env))
        except OSError as exc:
            self.log(f"report not written: {exc}")
            return None

    def exit_reason(self) -> str:
        """i18n key explaining why the engine ended on its own (by its exit code / last lines)."""
        text = " ".join(self._engine_lines)
        code = self.engine.exit_code if self.last_exit is None else self.last_exit
        if code == 4 or "BETA_EXPIRED" in text:
            return "engine_expired"
        if code == 5 or "HEADS_" in text:
            return "engine_no_heads"
        if code == 6 or "WEAK_GPU" in text:
            return "engine_weak_gpu"
        if "NO_ENCODER" in text:
            return "engine_no_encoder"
        if self.link.stopped == "input_failed" or code == 7:
            return "engine_port_busy" if self.link.reason == "port_busy" else "engine_input_failed"
        if self.link.stopped == "platform_failed" or code == 8:
            return "engine_platform_failed"
        if self.link.stopped == "input_ended":
            return "engine_input_ended"
        return "engine_exit"

    def stop(self) -> None:
        self.engine.stop()
        self.close_out()

    def last_report(self) -> Path | None:
        return engine.last_report(paths.reports_dir(self.env))


def env_view(env: Mapping[str, str]) -> dict[str, str]:  # used by tests and the diagnostics line
    return {k: v for k, v in env.items() if k in ("BACKEND", "ONNX_DIR", "HEAD_DIR", "OBS_STATE", "TRT", "TRT_BUILD", "ENCODER")}
