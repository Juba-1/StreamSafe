# StreamSafe engine

Covers risky characters in a game stream **before** viewers see them. The picture is held back by a
fixed delay (2 s by default); during that time every frame is analysed and each person who must be
covered gets an opaque garment that follows them. If anything goes wrong the viewer gets a
full-screen "back in a moment" slate - never an unprotected frame.

## How it fits together

```
 source ──► capture ──► AnalyzerWorker (own process) ─── decisions ──┐
 (file,                │   detect ─► eye ─► tracks ─► risk ─► pose ─► garments
  SRT/RTMP,            │                                              ▼
  OBS Virtual Camera)  └──► delay buffer ──────────────────────► OutputGate ──► sink
                                                                (fail closed)    (file, RTMP/SRT,
                                                                                  window for OBS)
```

| Package | Responsibility |
|---|---|
| `streamsafe/config.py` | every setting, grouped by component; each one can be set from the environment |
| `streamsafe/types.py` | data passed between stages: `Person`, `Layer`, `Decision`, `MotionRef` |
| `streamsafe/models/` | model protocols (`base.py`) + the PyTorch backend; risk heads (`heads.py`) |
| `streamsafe/analysis/` | `tracks.py` (who is who), `risk.py` (who must be covered), `analyzer.py` (one decision) |
| `streamsafe/garment/` | `shapes.py` (where the garment goes), `fabric.py` (how it looks), `builder.py` (one layer per person) |
| `streamsafe/render/` | `compositor.py` (which decisions protect a frame, drawing), `motion.py`, `slate.py` |
| `streamsafe/runtime/` | `runner.py` (realtime / offline), `worker.py` (analyzer process), `gate.py` (fail-closed output), `cuts.py`, `report.py` |
| `streamsafe/io/` | `sources.py` (file, FFmpeg URL, capture device), `sinks.py` (file, FFmpeg stream, window) |
| `bench/` | GPU benchmark tools: metrics, side-by-side videos, frame strips |
| `tests/` | stand-in models, end-to-end, failure injection, I/O, bit-exact equivalence with the previous engine |

Rules the code keeps:
- **Fail closed.** No decision for a frame, a failed decision, a scene cut with no fresh decision, a
  crashed analyzer: the gate repeats the last protected frame for at most 2 frames, then shows the slate.
- **Decisions are bound to frames and shots.** A frame is drawn only with decisions at most 18 frames
  away and never across a scene cut.
- **Models are sensors; the decision is code.** `analysis/risk.py` is plain, deterministic logic.
- **Opaque covers only.** No blur, no pixelation.
- **Swappable models.** The engine only talks to the protocols in `models/base.py`.

## Run

```bash
pip install -e ".[models]"          # engine + PyTorch models (GPU recommended)

# deterministic benchmark of a file (same result on any machine)
python -m streamsafe --input trailer.mp4 --output review.mp4 --mode offline --head-dir heads/

# realtime, file paced like a live feed
python -m streamsafe --input trailer.mp4 --output review.mp4 --head-dir heads/ --stats stats.json
```

### Connecting OBS

| Way | OBS side | StreamSafe | Restart of the stream needed? |
|---|---|---|---|
| **A. Virtual camera in, window out** | Virtual Camera -> output type *Source* = the game capture; add a *Window Capture* of "StreamSafe Output" and hide the raw game capture; delay game audio with *Sync Offset* | `--input "device:OBS Virtual Camera" --output window` | No (all steps can be done while live; to be confirmed on a real setup) |
| **B. SRT in, platform out** | Stream to `srt://127.0.0.1:9000` (Custom service) | `--input "srt://127.0.0.1:9000?mode=listener" --output "rtmps://<ingest>/<key>"` | Yes (OBS's stream target changes) |
| **C. OBS plugin** | planned (fork of the MIT obs-broadcast-delay / censure plugins) | - | Yes (OBS loads plugins at start) |

Audio is not processed by the engine yet: in A it is delayed inside OBS; B passes video only.

## Settings

All in `streamsafe/config.py`, e.g. `DELAY=1.5`, `EYE_EVERY=1`, `COLOR=black`, `LOGO=sponsor.png`,
`MAX_GAP=18`. CLI flags override the environment.

## Tests

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest                       # ~3 min on 2 CPU cores; no GPU or model download needed
```

`tests/test_equivalence.py` runs the previous engine (`engine2_ref`, set `STREAMSAFE_REF`) next to
this one and requires identical decisions, garment positions and output pixels.
