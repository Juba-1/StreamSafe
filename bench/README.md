# bench/ - measure and review the engine

Tools that run the engine's output against video and report numbers or make review material. None of them
change the engine. Run them from the repository root as modules (`python -m bench.<name> ...`); each file's
docstring has its exact usage.

| Tool | Question it answers | Input -> output |
|---|---|---|
| `diag_metrics` | How good is the cover? Coverage, lag behind the person, shaking (jitter p50/p90/p99, pops per 10 s), ghosts, size jumps, late covers | source video + one or more `--diag` logs -> JSON line per log |
| `disagree` | Where do two head sets / configs disagree? | video + two `--diag` logs -> contact sheet (jpg) |
| `why_covered` | Why is this person covered? The biggest person's score, beach prior and threshold, per decision | `--diag` log -> table |
| `sbs` | Before vs now, side by side | two videos -> 1920x540 H.264 |
| `strips` | Frame-by-frame look at moments of a side-by-side | sbs video + times -> jpgs |
| `mux_audio` | Review copy with sound, silenced where the engine muted it | protected video + original + stats -> mp4 |
| `summ` | One summary line per engine run | engine stdout -> text |
| `soak` | Does the live path stay healthy for hours? Late / dropped frames, memory per process | clips + minutes -> soak report |
| `onnx_speed` | How fast is one exported model on this GPU, per batch size (first run, steady, after a size change, bound to the GPU)? | one `.onnx` file -> JSON line |
| `eye_equiv` | Does the eye still score as before (after a dependency or revision change)? | head dir + reference crops -> pass/fail |

Typical review loop (GPU pod):

```bash
python -m streamsafe --input clip.mp4 --output now.mp4 --mode offline --head-dir heads/ --diag now.json
python -m bench.diag_metrics clip.mp4 75 now=now.json
python -m bench.sbs before.mp4 now.mp4 sbs.mp4
```

`labels/` holds Ali's label CSVs used by `training/` (kept here for history; never edit by hand).
