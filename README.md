# StreamSafe engine

Covers risky characters in a game stream **before** viewers see them. The picture is held back by a fixed
delay (2 s by default). During that time every frame is analysed, and each person who must be covered gets an
opaque garment that is drawn once and then moves with them. Explicit scenes get a whole-screen card and the
game audio is muted. If anything goes wrong, viewers get a full-screen slate, never an unprotected frame.

Target setup: the game runs on a console, a capture card feeds the streamer's PC, OBS streams, and StreamSafe
sits between the capture and the stream scene (`docs/OBS_SETUP.md`).

## How it fits together

```
 source ──► capture ──► AnalyzerWorker (own process) ───── decisions ─────┐
 (file,                │   detect ─► track ─► eye ─► risk ─► pose ─► garment │
  SRT/RTMP,            │   scene model + HUD reader ─► scene layer           ▼
  OBS Virtual Camera)  └──► delay buffer ─────────────────────────────► OutputGate ──► sink
                                                                     (fail closed,    (file, RTMP/SRT,
                                                                      steady placement) window for OBS)
                                                                          │
                                                          OBS state file ─┴─► obs/streamsafe_guard.lua
                                                          (heartbeat + mute)   (slate + game-audio mute)
```

One decision per analysed frame says who is covered and with which garment shape. The gate draws every
output frame from the decisions next to it (same shot only) and moves each garment with its person.

| Folder | Responsibility |
|---|---|
| `streamsafe/config.py` | every setting with its default and the reason for it; each one can be set from the environment |
| `streamsafe/types.py` | data passed between stages: `Person`, `Layer`, `Decision`, `MotionRef` |
| `streamsafe/models/` | model protocols (`base.py`), risk heads (`heads.py`), HUD reader (`ocr_hud.py`), PyTorch backend |
| `streamsafe/analysis/` | `tracks.py` who is who, `risk.py` who must be covered, `scene.py` whole-screen cover, `analyzer.py` one decision |
| `streamsafe/garment/` | `shapes.py` where the garment goes, `fabric.py` how it looks (colour, logo), `builder.py` one layer per person |
| `streamsafe/render/` | `compositor.py` drawing, `motion.py` following, `smooth.py` steady placement, `slate.py` slate and scene card |
| `streamsafe/runtime/` | `runner.py` realtime/offline loop, `worker.py` analyzer process, `gate.py` fail-closed output, `cuts.py`, `report.py` |
| `streamsafe/io/` | `sources.py` inputs, `sinks.py` outputs, `obs_link.py` state file for the OBS guard |
| `obs/` | `streamsafe_guard.lua`: OBS script that shows the slate and mutes game audio when the engine says so or stops |
| `bench/` | measure and review the engine (metrics, side-by-side videos, soak test). See `bench/README.md` |
| `training/` | build the eye's heads and the scene head from labeled crops. See `training/README.md` |
| `experiments/` | one-off research scripts behind past decisions. Not used by the engine. See `experiments/README.md` |
| `tests/` | stand-in models: end-to-end, failure injection, I/O, OBS guard, equivalence with the previous engine |

## What protects the stream

| Layer | What it does | Where |
|---|---|---|
| Person garments | Each risky person (the eye's score above the threshold) gets an opaque garment shaped by their pose. Small people get a plain full cover. | `analysis/risk.py`, `garment/` |
| Release rule | A person is uncovered only after 1 s of clear evidence. A big person seen for the first time needs a stronger score to be covered (no flash covers on faces). | `analysis/risk.py` |
| Following | Garments follow their person between decisions (motion estimate), are steadied by a small per-person filter, and fast movers get a margin in their direction of motion. | `render/motion.py`, `render/smooth.py`, `garment/builder.py` |
| Scene layer | Explicit scenes (trained scene head, HUD prompts): whole-screen card + game-audio mute, with lead and tail times. | `analysis/scene.py` |
| Fail closed | No decision for a frame, a failed or stale decision, a cut with no fresh decision, decisions too far apart (a slow GPU, `max_span`), a crashed analyzer: the last protected frame is repeated for at most 2 frames, then the slate. A crashed or failing analyzer is restarted (slate while it loads). | `runtime/gate.py`, `runtime/runner.py` |
| Bounded latency | A frame more than 0.5 s late is dropped, never shown late, so delay and memory cannot grow. | `runtime/runner.py` |
| OBS guard | Slate + game-audio mute when: no heartbeat for 0.5 s, the captured output window is empty, a named game-audio source is missing, or the streamer presses the PANIC hotkey. The raw game source on air is switched back to the protected scene. | `obs/streamsafe_guard.lua` |
| Brand | Garment colour and chest logo, slate and scene-card image/colour/text are streamer settings. | `config.py` (`GarmentConfig`, `BrandConfig`) |

Rules the code keeps:
- **Fail closed.** Doubt, error or a missing decision means cover, never show.
- **Decisions are bound to frames and shots.** A frame is drawn only with decisions at most 18 frames away and never across a cut.
- **Models are sensors; the decision is code.** `analysis/risk.py` and `analysis/scene.py` are plain, deterministic logic.
- **Opaque covers only.** No blur, no pixelation. Audio is muted with every whole-screen cover.
- **Swappable models.** The engine only talks to the protocols in `models/base.py`.

## Models

Two backends run the same models (`BACKEND`): `torch` (PyTorch + CUDA, NVIDIA only: research, benchmarks,
export) and `onnx` (ONNX Runtime: NVIDIA via CUDA, any DirectX 12 GPU on Windows via DirectML; the product path).
`tools/export_onnx.py` writes the ONNX folder with a `manifest.json` (sha256 + pins) that the engine checks before
loading anything. Without a GPU the onnx backend refuses to start (never a silent CPU fallback).

All are downloaded from the Hugging Face Hub at a **pinned commit** (`*_revision` in `config.py`). Nothing new
reaches a streamer's PC until someone moves the pin on purpose.

| Role | Model | License |
|---|---|---|
| Person detector + mask | RF-DETR-Seg Nano (`rfdetr` package) | Apache-2.0 |
| Eye (who is risky) | `nvidia/C-RADIOv4-H` + our gated logistic heads (`HEAD_DIR`) | NVIDIA Open Model License (heads: ours) |
| Scene model + beach prior | `google/siglip2-base-patch16-224` + our scene head | Apache-2.0 |
| Pose (garment shape) | `usyd-community/vitpose-base-simple` | Apache-2.0 |
| HUD reader (optional) | RapidOCR via `rapidocr_onnxruntime==1.2.3` | Apache-2.0 |

Not allowed in the product: AGPL code or weights (Ultralytics YOLO, NudeNet), research-only weights.

## Run

```bash
pip install -e ".[models]"          # engine + PyTorch models (NVIDIA GPU)

# deterministic benchmark of a file (same result on any machine)
python -m streamsafe --input trailer.mp4 --output review.mp4 --mode offline --head-dir heads/

# realtime, file paced like a live feed
python -m streamsafe --input trailer.mp4 --output review.mp4 --head-dir heads/ --stats stats.json
```

Live with OBS (console + capture card): follow `docs/OBS_SETUP.md`. In short, OBS Virtual Camera (hidden raw
scene) goes in, `--output window` comes out, and OBS captures that window. The guard script handles the slate and
mutes the audio. The other way is SRT in, platform out:
`--input "srt://127.0.0.1:9000?mode=listener" --output "rtmps://<ingest>/<key>"` (video only).

## Settings

Every setting lives in `streamsafe/config.py`, grouped by component, with a comment saying why it has that value.
Set any of them from the environment by its upper-case name. CLI flags override the environment. Examples:

| Setting | Example | Meaning |
|---|---|---|
| `DELAY` | `DELAY=2.0` | stream delay in seconds |
| `COLOR` / `LOGO` | `COLOR=#2c2c2c LOGO=sponsor.png` | garment colour (palette name or hex) and chest logo |
| `SLATE_IMAGE` / `SLATE_COLOR` / `SLATE_TEXT` | `SLATE_IMAGE=logo.png` | the fail-closed slate |
| `SCENE_IMAGE` / `SCENE_TEXT` | | the explicit-scene card |
| `HUD_OCR` | `HUD_OCR=1` | read the game's on-screen prompts (needs `rapidocr_onnxruntime==1.2.3`) |
| `SMOOTH`, `MOTION_MARGIN` | `SMOOTH=0` | steady placement and fast-mover margin (both on by default) |
| `BACKEND` / `ONNX_DIR` | `BACKEND=onnx ONNX_DIR=C:\StreamSafe\models` | model backend and the exported model folder |
| `OBS_STATE` | `OBS_STATE=C:\ProgramData\StreamSafe\state.txt` | state file read by the OBS guard script (ASCII path) |

## Security notes

- Model code: only the eye runs repository code (`trust_remote_code`), and only at its pinned commit. All
  third-party model downloads are pinned to a commit.
- The engine never sees the stream key in the recommended OBS setup. OBS keeps it.
- External programs (FFmpeg) are started with an argument list, never through a shell.
- The OBS state file holds one line (alive, frame counter, mute, full-screen), written atomically. No network, no password.
- Offline tools that read untrusted files (videos, label sheets) only decode them and never run them.

## Tests

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest                       # ~4 min on 2 CPU cores; no GPU or model download needed
```

`tests/test_equivalence.py` runs the previous engine (`engine2_ref`, set `STREAMSAFE_REF`) next to this one with
steadying and the motion margin off (engine2 had neither) and requires identical decisions and output pixels.
