# Security and antivirus posture of the Windows beta

What the installer and the app do on a streamer's PC, so that a person, a SmartScreen prompt or an antivirus
engine finds no surprise. Claims are labelled **TESTED** (a test or the Windows CI smoke run checks it),
**SOURCE-READ** (read in the code, not executed on Windows CI) or **DOC-CLAIM** (a vendor's statement we did not verify).

## What is installed, and where

| Item | Where | Claim |
|---|---|---|
| Program folder (two exe + libraries + FFmpeg + risk heads + licences + OBS guard script) | `%LOCALAPPDATA%\Programs\<name>` | TESTED (installer smoke) |
| Start-menu shortcut, uninstaller, uninstall entry | per user: Start menu of the user, `HKCU` only | TESTED |
| The user's data (models, engines, reports, settings, protected stream key) | `%LOCALAPPDATA%\<name>` | TESTED (path), SOURCE-READ (what is stored) |
| OBS heartbeat file | `%LOCALAPPDATA%\<name>\obs\state.txt` (8.3 short form, else `C:\Users\Public\<name>`, for a user name with non-ASCII letters) | TESTED (logic, with a fake short-path function) |

The installer asks for **no administrator rights** (`PrivilegesRequired=lowest`) and the smoke test checks that nothing
was written to `HKLM`, Program Files or ProgramData. **TESTED** (the runner is an administrator account, so this checks
what was written, not that the installer would have been refused elevation).

## What it does not do

* No service, no driver, no kernel component. **TESTED** (no service; no scheduled task).
* No autostart: no `Run` key, no startup shortcut, no scheduled task. **TESTED**
* No hidden or detached processes. The window starts the engine as an ordinary child process (no console window), and the
  engine ends when the window's pipe closes, so no engine is left running without its window. A clean stop writes the
  session report; a stop that does not end in 20 s kills the engine's process tree. **TESTED** (fake-engine tests: start,
  stop, crash; smoke: no process left after the window closes)
* No self-modifying code, no code download, no code from the network: the app and the installer are replaced only by a
  new installer the streamer runs. Downloaded files are data (model files, risk-head text files) and vendor libraries
  named below. The update check can announce a new app version as text; it never downloads or runs it. **SOURCE-READ**
  (`streamsafe/app/updates.py` handles data files and a version string only), **TESTED** (signature, replay, URL checks)
* No elevation prompts, no UAC manifest requesting admin, no packers or obfuscation: PyInstaller **one-folder** (not
  one-file: nothing unpacks itself into a temp folder at every start). The exe carry a version resource (company, product,
  version). **TESTED**
* No telemetry. The app sends nothing about the user or the PC; the only requests are the ones listed next.
  **SOURCE-READ** (`release.allowed_hosts()` lists every address; the User-Agent is `StreamSafe-app` without a version).

## Network: every connection the app makes

| When | To | What | Safeguards |
|---|---|---|---|
| First run, user presses "Download models" | `release.MODEL_BASE_URL` (Hugging Face `juba-1/streamsafe-models`, folder `v1/`) | `manifest.json`, then the files it lists | https only (redirects too); each file sha256 + size against the manifest; the whole set is verified again (as the engine does at every start) before an atomic swap; the pack in use is kept as `models.prev`. **TESTED** |
| Start of the window, only if `release.UPDATE_MANIFEST_URL` and a public key are set | the update address (empty by default: no request) | one small JSON manifest | ed25519 signature checked against the key built into the app; a lower serial than the last one seen is ignored (no replay of old models); size limited to 256 KB; any failure: carry on with what is installed. **TESTED** |
| User presses "Update now" | URLs inside the signed manifest | new model files and risk-head files (data) | https only; sha256 + size per file from the signed manifest; heads must load; atomic swap, previous kept for rollback. **TESTED** |
| Only if the user says yes to the fast NVIDIA mode | `files.pythonhosted.org` and `pypi.nvidia.com` | ONNX Runtime GPU, CUDA, cuDNN, TensorRT wheels | pinned by sha256 (`release.NVIDIA_WHEELS`); unpacked with path / link / size checks into the user's data folder; sealed by `READY.json`; the app uses it only if sealed from this build's pins. **TESTED** (download logic with local server), see "NVIDIA" below |
| During a stream | the address the streamer typed | the video | the stream key is stored only protected with DPAPI (when the user ticks "remember"), is passed to the engine in the child's **environment** (never on a command line, so it is not in the process list) and is blanked out of every line the window shows. **TESTED** (fake protector; fake engine) |

A model repository that is private or missing is reported plainly (401 / 403 / 404 / no connection) and the user can pick
a zip or folder we sent them instead; the same checks apply to it. **TESTED**

## Fail closed

* A beta build refuses to start a stream after its build-time expiry date (default 30 days) and says to download the new
  version; a damaged date counts as expired. A stream already running is not cut. **TESTED** (unit + `run` exit code 4)
  A clock set back defeats this; it is a beta courtesy, not a licence check.
* The engine refuses to start without models that pass their sha256 check, without usable risk heads (exit code 5) or
  without a working H.264 encoder (a network stream). A test build with placeholder heads (CI) never streams. **TESTED**
* Whatever the window cannot confirm is shown as covered or stopped, never as "protected". **TESTED**

## Third-party code

FFmpeg is an **LGPL** build (shared DLLs in `ffmpeg\bin`, replaceable), pinned by sha256 in `installer/ffmpeg.lock.json`; the
smoke test refuses a build with `--enable-gpl`, `--enable-nonfree` or libx264. **TESTED** Licence texts and the NVIDIA
Open Model License attribution are installed in `licenses\`. NVIDIA's CUDA / cuDNN / TensorRT are never shipped; see
`docs/NVIDIA_TRT.md` for the licence question.

## Code signing and false positives

The beta installer is **not code-signed** (no certificate yet): Windows SmartScreen will warn "unknown publisher", and some
antivirus engines score an unsigned, newly built PyInstaller program higher than average. What lowers the risk without a
certificate: one-folder build, no packer, version resource, per-user install, no network except the table above. A
certificate (OV or EV) and submitting each release to Microsoft's and the major vendors' false-positive portals are the
real fix and an owner decision. **DOC-CLAIM** (PyInstaller bootloader signatures are a known source of false positives;
we have not scanned the artifact with any vendor).
