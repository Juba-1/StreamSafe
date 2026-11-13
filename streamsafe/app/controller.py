"""What the window does, without any window: the settings and the stream key, the system check, installing the
models, the NVIDIA fast mode, the update check, and starting / watching / stopping the engine.

``ui.py`` only draws this and calls it. Everything here is testable without a screen, and every failure ends on the
safe side: no start when anything is unclear, and a running engine that stops or stalls is shown as covered.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

from streamsafe.app import engine, fetch, heads, keystore, modelpack, nvidia_pack, paths, release, updates
from streamsafe.app import settings as settings_mod
from streamsafe.app.settings import OUTPUT_RTMP, Settings
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
        self.settings: Settings = settings_mod.load(paths.settings_file(self.env))
        self.keys = keystore.KeyStore(paths.key_file(self.env), keystore.default_protector() if protector is None else protector)
        self.engine = engine.EngineProcess(self._engine_line, engine.Redactor())
        self.tracker = engine.StatusTracker()
        self.check_result = CheckResult()
        self.use_nvidia = False  # decided by the last check: engines ready AND the NVIDIA libraries in place
        self.last_exit: int | None = None
        self.update: updates.Update | None = None
        self._engine_lines: list[str] = []

    # --- settings and key ---------------------------------------------------------------------------------
    def save_settings(self) -> None:
        try:
            settings_mod.save(paths.settings_file(self.env), self.settings)
        except OSError as exc:
            self.log(f"settings not saved: {exc}")

    def stored_key(self) -> str:
        return self.keys.load() if self.settings.remember_key else ""

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

    def blockers(self, key: str = "") -> list[str]:
        """i18n keys (``why_*`` / ``bad_*``) of what stops a start. Empty: it may start."""
        out: list[str] = []
        if self.engine.running:
            return ["why_running"]
        if self.expired():
            out.append("why_expired")
        out += self.settings.problems()
        if self.settings.output == OUTPUT_RTMP and not settings_mod.valid_key(key):
            out.append("bad_key")
        if self.models_problem():
            out.append("why_models")
        why = self.heads_problem()
        if why:
            out.append("why_heads_placeholder" if why == "placeholder" else "why_heads")
        if self.check_result.verdict == "fail":
            out.append("why_check")
        return out

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

    def can_roll_back(self) -> bool:
        return modelpack.previous(paths.models_dir(self.env)).is_dir()

    def roll_back(self) -> None:
        modelpack.rollback(paths.models_dir(self.env))

    # --- the engine -----------------------------------------------------------------------------------------
    def _engine_line(self, line: str) -> None:
        self._engine_lines = [*self._engine_lines[-30:], line]
        self.log(line)

    def start(self, key: str) -> None:
        """Start the engine. Raises RuntimeError with the blocking i18n key when it must not start."""
        why = self.blockers(key)
        if why:
            raise RuntimeError(why[0])
        s = self.settings
        url = engine.output_url(s.rtmp_url, key) if s.output == OUTPUT_RTMP else ""
        env = engine.engine_env(os.environ, self.use_nvidia, url, self.env, translate=s.translate_ar and not self.translate_problem())
        state = paths.obs_state_file(self.env)
        state.unlink(missing_ok=True)  # a line left by an earlier run must not look like this run's
        state.parent.mkdir(parents=True, exist_ok=True)
        self.engine.redactor_secrets(key, url)
        self.tracker.reset()
        self.last_exit = None
        self._engine_lines = []
        self.engine.start(engine.exe_command(engine.run_args(s, paths.reports_dir(self.env))), env)

    def status(self) -> str:
        running = self.engine.running
        if not running and self.engine.exit_code is not None and not self.engine.stop_requested:
            self.last_exit = self.engine.exit_code
        failed = self.last_exit not in (None, 0)
        return self.tracker.update(running, engine.read_state(paths.obs_state_file(self.env)), failed)

    def exit_reason(self) -> str:
        """i18n key explaining why the engine ended on its own (by its exit code / last lines)."""
        text = " ".join(self._engine_lines)
        if self.last_exit == 4 or "BETA_EXPIRED" in text:
            return "engine_expired"
        if self.last_exit == 5 or "HEADS_" in text:
            return "engine_no_heads"
        if "NO_ENCODER" in text:
            return "engine_no_encoder"
        return "engine_exit"

    def stop(self) -> None:
        self.engine.stop()

    def last_report(self) -> Path | None:
        return engine.last_report(paths.reports_dir(self.env))


def env_view(env: Mapping[str, str]) -> dict[str, str]:  # used by tests and the diagnostics line
    return {k: v for k, v in env.items() if k in ("BACKEND", "ONNX_DIR", "HEAD_DIR", "OBS_STATE", "TRT", "TRT_BUILD", "ENCODER")}
