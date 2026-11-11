# Speed plan: 10 decisions/s on an RTX 3060 / 4060

Branch `speed`, 10 Nov 2026. Goal: the product path (`BACKEND=onnx`) holds 10 decisions/s on consumer GPUs
without any decision becoming less protective and without touching the fail-closed rules.

Claim labels: **TESTED** (ran here, CPU, stand-in models; or the GPU run named next to it), **SOURCE-READ**
(read in this repository or in the named package), **DOC-CLAIM** (vendor documentation or model hub),
**ESTIMATE** (arithmetic from the run 55 / 57 numbers, not measured), **UNKNOWN** (needs the GPU run named
next to it).

Update 11 Nov 2026, after run 57 (RTX 3090): TensorRT is measured and is now the NVIDIA product path.
Sections 1, 2, 3.3 and 4 carry the numbers; 3.5 (realtime headroom) and E8 / E9 are new. Installer-facing
details: `docs/NVIDIA_TRT.md`.

## 1. Where the time goes

Run 55, RTX A6000, volleyball clip (crowds), ms per decision:

| | total | eye | detector | pose | rest (CPU) |
|---|---|---|---|---|---|
| torch, eye bf16 | 94 | 46 | 27 | 9 | ~12 |
| ONNX fp32 | 133 (p95 180) | 96 (p95 140) | 20 | 7 | ~10 |

Run 57, RTX 3090, same clip, mean ms per decision (TESTED):

| | total | eye | detector | pose |
|---|---|---|---|---|
| torch | 124 | 66 | | |
| ONNX CUDA fp32 | 168 | 126 | | |
| ONNX TensorRT fp16 | 74 | 41 | 18 | 5 |

- TESTED (run 57): one eye crop costs 19 to 26 ms on CUDA fp32 and 5.2 to 8.3 ms on TensorRT fp16. Realtime on
  CUDA fp32 the 3090 holds 5.4 decisions/s of the 10 wanted; TensorRT brings the mean decision under one step.
- The eye is 72 % of an ONNX decision and all of the gap to torch. Nothing else matters until it is fixed.
- ESTIMATE: ViT-H/16 at 384x224 is 336 patch tokens x 32 blocks, about 0.2 TMAC per crop. 96 ms fp32 is then
  about 7 crops per decision on average (about 14 ms per crop fp32, about 6.5 ms in bf16). UNKNOWN: the real
  count is the `embedded` entry of the run 55 stats JSON; read it before trusting any number below.
- ESTIMATE: an RTX 3060 has about a third of the A6000's throughput, an RTX 4060 about 40 % (vendor fp32
  figures; DOC-CLAIM from memory, to be replaced by E5). The same crowd decision in ONNX fp32 is then about
  400 ms on a 3060: 2.5 decisions/s, and over the `max_span` limit (9 frames = 300 ms), so the slate.
- SOURCE-READ: the slate comes from single slow decisions, not from the average. Two things make a decision
  slow: (a) after a cut everybody is new, and every new person gets two looks in a row (`n_obs >= 2` in
  `Tracker.motion_hints`); (b) people followed by motion were all looked at in the same decision, so they all
  fall due again in the same decision, every `eye_every` decisions. TESTED on stand-in models: 9 still people
  give eye batches of 9, 9, 0, 0, 9, 0, 0, 9 (`tests/test_eye_budget.py`).
- TESTED (this 2-core CPU): cutting and resizing one eye crop costs 0.7 to 2.0 ms of CPU (PIL bicubic), so
  about 10 % of the eye time once the eye is fp16. It is exact by design (`eye_common.prepare`) and stays.

Budget on a 3060, ESTIMATE: detector about 60 ms and pose about 20 ms in ONNX fp32 leave nothing of a 100 ms
decision. With fp16 or TensorRT for both (about 25 + 8 ms) and 10 ms of CPU, about 55 ms remain for the eye:
2 to 3 crops at about 20 ms per fp16 crop. To stay under `max_span` at all the eye may use about 130 ms: 6 crops.
So on a 3060 the big eye can hold 10/s for a few people and "no slate" for a crowd, but not 10/s for a crowd.
That last step needs a smaller eye (section 5).

## 2. The levers, ranked by gain per risk and effort

| # | Lever | Expected gain | Risk | Effort | Status |
|---|---|---|---|---|---|
| 1 | fp16 eye (`EYE_FP16=1`) | eye about 2x (96 -> about 48 ms on the A6000), 1.3 GB less VRAM. ESTIMATE from torch bf16 vs ONNX fp32 on the same card | low: fp16 keeps more precision than the bf16 torch path that all heads were validated on | one GPU run | export fixed in 9c75c9e (fp16 from PyTorch on the GPU); matters for CUDA / DirectML only, TensorRT makes its own fp16 from `eye.onnx`: E1 |
| 2 | Warm every batch size at start | removes the first-sight pause per batch size. DOC-CLAIM: ONNX Runtime's CUDA provider searches convolution algorithms exhaustively per new input shape by default. Size UNKNOWN (E2 `first_ms`) | none: start-up only | done | implemented |
| 3 | Eye budget: at most K crops per decision, looks spread over decisions | bounds the eye time of every decision at K crops, so crowd decisions stop crossing `max_span`; a still crowd of N drops from peaks of N to N/3 per decision at the same average (TESTED on stand-ins) | medium: people the eye has not seen yet get a plain cover (more covers, see 3.2); look timing shifts by up to 2 decisions | done, off by default; can follow the measured device (3.5) | implemented, needs E4 / E9 |
| 4 | TensorRT provider (`TRT=1`) | TESTED (run 57, 3090): decision 168 -> 74 ms against CUDA fp32, eye crop 19-26 -> 5.2-8.3 ms; faster than torch (124 ms) | medium, operational: NVIDIA only, 2.5 GB of extra libraries (SOURCE-READ, `docs/NVIDIA_TRT.md`), about 3 minutes of engine build per card and driver, fp16 agreement (TESTED on the trailer: eye only_torch = 0) | done: safe engine cache, fallback to CUDA, `prepare` step (3.3) | **the NVIDIA product path**; E8 before shipping |
| 5 | Our own small eye (distilled, section 5) | 7x to 30x fewer operations per crop; the only lever that reaches 10/s in a crowd on a 3060 and the only one that helps AMD / Intel as much as NVIDIA | high: accuracy on the hard cases must be proven on Ali's labels | 1 to 2 weeks, GPU rent | proposal |
| 6 | Smaller eye input for small people | 1.4x (320x192) to 2.3x (256x144) on those crops | medium: run 36 showed detail matters (women 63 -> 77 % with full-resolution crops); heads must be retrained per size | days, GPU + labels | not started; overtaken by 5 if 5 works |
| 7 | Fixed batch sizes (padding) | only if the runtime prepares again on every size change (DirectML is the suspect). UNKNOWN: E2 `switch_extra_ms`. Not needed with TensorRT: one engine covers batch 1 to 16 through the shape profile | low | small | measure first |
| 8 | CUDA graphs | DOC-CLAIM: needs I/O binding, fixed shapes and every node on CUDA. Helps models made of many small kernels (detector, pose), hardly a ViT-H. TensorRT covers the same ground | medium | medium | not planned |
| 9 | Detector and eye in a pipeline across decisions | both use the same GPU, so only the CPU parts overlap: at most 10 to 15 %. Adds a decision of latency and threads inside the fail-closed core | medium | medium | not planned |
| 10 | I/O binding, crops kept on the GPU | the copy is 16.5 MB for 16 crops: 1 to 3 ms. Resizing on the GPU would change the scores (`eye_common.prepare` says so) | low gain | medium | measure with E2 `bound_p50`, then drop |
| 11 | Reuse the scene frame's features | none: the scene model is SigLIP2 on a 224 px frame, another embedding space. Running the eye once on the whole frame would give up full-resolution crops (run 36) and need new heads | high | high | rejected |

Fewer eye calls beyond lever 3, considered and not built:
- Skipping looks at people who "did not change" (thumbnail difference): a look saved on an uncovered person is
  only safe if the cheap test never misses a change. That needs data first (E7), not code.
- Looking at covered people less often than every `eye_every` decisions all the time: the budget already does
  this when it is short of crops, and only then.

## 3. What is implemented on this branch

Default behaviour is unchanged: with `EYE_BUDGET=0` and `TRT=0` the engine makes the same decisions and draws the
same pixels as before (TESTED: `tests/test_equivalence.py` against `engine2_ref`, and the whole suite).

### 3.1 Warm-up of every batch size (`Analyzer.warmup`)
The eye is run once for every batch size 1..16 (or 1..K with a budget), the pose model for 1..8, before the
analyzer reports ready. Why safe: start-up only, blank frames, no state.

### 3.2 Eye budget (`EYE_BUDGET=K`, `EYE_MAX_AGE`; `streamsafe/analysis/eye_schedule.py`)
One rule: **at most K crops per decision, and whoever needed a look and did not get one is covered until the
eye has seen them.**

- Order of looks: biggest people first (risk register #10). Covered people whose look falls due come last,
  because waiting changes nothing on screen for them, until they have waited `EYE_MAX_AGE` (9) decisions.
- Put off and followed by motion: the person keeps the track. A covered person stays covered; an uncovered
  one gets a plain cover. In both cases the risk policy does nothing for that decision: no evidence, no
  observation counted, and the release clock stands still (`risk.py`).
- Put off and unknown to the tracker (new, moved a lot, after a cut): plain cover under a waiting id, no track.
- When the look comes, evidence toward covering counts once per decision waited (`Person.waited`), so a cover
  by accumulated evidence does not come later than without the wait. Evidence toward release counts once.
- Early looks: spare budget goes to uncovered people with the oldest reading, up to the crowd's average need
  (people / `eye_every`), so looks do not all fall due together. Covered people are never looked at early.

Why it is safe, and where the claim ends:
- TESTED: never more than K crops in a decision; a risky person is covered in every decision from the first,
  seen or not; a person who turns risky is covered within `eye_every - 1` decisions with and without a budget;
  decisions without a look never count toward a release; without a look there is no release at all; through
  the runner and the gate (offline and realtime) the walker stays covered in every frame.
- SOURCE-READ: `max_span`, `max_gap`, `hold`, the slate and the worker are untouched.
- Not claimed: identical decisions with and without a budget. Looks happen in other decisions, so a cover can
  start or end up to `eye_every - 1` decisions earlier or later than in the unbudgeted run. The bound on how old
  a reading may be (`eye_every - 1` decisions, or covered) is the same as today. E4 measures the difference.
- The price is more covers: after a cut into a crowd larger than K, the smaller people wear a plain cover for
  a few decisions (TESTED on stand-ins: 9 people, K=3: from the 4th decision on only the risky are covered). A crowd larger than about
  3K never fits: its smallest people stay covered. Ali should see this on real clips (E4 side by side) before
  it is switched on; the alternative today in the same moments is the full-screen slate.

### 3.3 TensorRT (`TRT=1`; `streamsafe/models/onnx_backend/session.py`, `engines.py`, `streamsafe/prepare.py`)
Decision after run 57: TensorRT is the NVIDIA product path. `TRT` stays off in `config.py` (the default must
start on any GPU); the NVIDIA edition of the app turns it on. Engines are built from the fp32 files with fp16 on.

TESTED on the GPU (run 57, RTX 3090, ONNX Runtime 1.22.0 + TensorRT 10.9): the numbers of section 1; build on
first start detector 77 s, pose 28 s, eye 58 s (the 2.6 GB `eye.onnx` parses); loading afterwards 6 to 14 s;
trailer: eye `only_torch` = 0; volleyball clip: 97 torch-only against 33 trt-only pairs, which look like
timing differences (to be confirmed cell by cell in E8).

What is new on this branch (all TESTED on CPU with ONNX Runtime and the PC faked, `tests/test_trt_engines.py`,
`tests/test_prepare.py`; nothing of it has run on a GPU yet: E8):
- **Safe engine cache.** Key = sha256 of model + weight file, name and driver version of every NVIDIA card,
  ONNX Runtime version, TensorRT version, fp16 flag, batch range and build options. The folder is named after
  the key and the full key is written into `seal.json` beside the engines, with size and sha256 of every file.
  A build goes into a private folder; the seal is written last and the folder then moved into place, so an
  interrupted build is never read. At load the key and every file are checked before TensorRT is asked. A new
  card, driver or library version, or a damaged file, means rebuild; the stale engine is not opened.
- **Fallback chain TensorRT -> CUDA -> refuse to start.** A model TensorRT cannot run (libraries missing, not
  prepared with `TRT_BUILD=0`, build failed, NaN in the engine's test run) runs on CUDA; the reason goes to
  stderr at start (`STREAMSAFE_DEVICE ...`) and into the run report (`summary.device_notes`; `summary.gpu`
  reads `onnx-trt+cuda` when only some models fell back). SOURCE-READ: ONNX Runtime 1.22.0 by itself retries a
  failed TensorRT session on the CPU alone and only prints the error; that session is never used.
- **`python -m streamsafe.prepare --models DIR`** builds or verifies every engine with progress lines and
  exits non-zero with the reason on failure; `python -m streamsafe.syscheck --models DIR --engines` reports
  "engines ready" as an optional check. The app runs it before every start and starts the engine with
  `TRT_BUILD=0`, so the minutes are paid behind a progress bar and never at "go live".
- **A model that answers NaN fails closed.** An fp16 engine that overflows returns NaN, and NaN compares false
  with every threshold: the detector would find nobody and the eye would call everybody safe. Every new engine
  must answer a test input with finite numbers before it is sealed, and at run time non-finite detector, eye or
  scene outputs fail the decision (slate) on every backend.
- The engine's batch range no longer follows `EYE_BUDGET` (it is part of the key: changing a setting must not
  cost a rebuild). It is fixed at 1 / 4 / 16, the range run 57 used.
- Shape profile: the eye and the pose model pass their batch range (1 to 16), so one engine covers every batch
  size. DOC-CLAIM: without a profile the engine is rebuilt whenever a new size falls outside the ones seen.

Still open:
- UNKNOWN (E8): that ONNX Runtime finds the engine again after the build folder was moved into place; start
  time with the sha256 check of the engine files; GPU memory during a build at engine start on an 8 GB card
  (`prepare` builds one model at a time); the scene model's build time; everything on Windows.
- The scene model gets an engine too (as in run 57's engine runs), although it runs twice a second and its fp16
  agreement was not reported. E8 reads `scene p_absdiff`; if it is not clean, the scene model goes back to CUDA.
- Licence of the TensorRT libraries inside our installer: for Ali, `docs/NVIDIA_TRT.md` section 1.

### 3.4 `bench/onnx_speed.py`
One model file, opened exactly as the engine opens it: per batch size the first run, p50 / p95, ms per item,
the same with input and output bound to the GPU, and the extra time after a batch-size change. It turns the
UNKNOWNs of levers 1, 2, 4, 7 and 10 into numbers. TESTED on CPU with a tiny model.

### 3.5 Realtime headroom and the automatic eye budget (`analysis/headroom.py`, `EYE_BUDGET_AUTO`)
The same card is fast or slow depending on the path, and the path can change under the streamer (TensorRT falls
back to CUDA). So the analyzer times its own models once, at the end of warm-up (detector, pose for 2 people,
the eye with 1 and with 16 crops; median of 3 runs each, on blank frames: a model's time does not depend on
the picture), adds 12 ms for the CPU work it cannot time, and gets `decision_ms(n)` for n eye crops.

Always on (it changes no decision): the numbers go into the run report (`summary.headroom`), and when an
ordinary decision (4 crops) does not fit into one step (100 ms) the worker prints `STREAMSAFE_SLOW ...` at
start. That is how a slow path is never taken silently: the device is named, the reason is named, the rate is named.

Optional, **off by default** (`EYE_BUDGET_AUTO=1`, realtime only, and only when `EYE_BUDGET=0`): the budget is
set once, before the first decision, by one rule:
- if a decision with a full eye batch (16) takes at most 0.6 of the slate limit (0.6 x 9 frames = 180 ms): no
  budget;
- otherwise the largest K whose decision fits into those 180 ms, never below 3.

On the run 57 numbers (TESTED as arithmetic, `tests/test_headroom.py`): TensorRT on the 3090 gives 118 ms for a
full batch, so no budget, with room to spare; CUDA fp32 on the same card gives 343 ms, so K = 7.

Why this rule and not another:
- The budget's price is extra plain covers (3.2), so it must stay off where it is not needed. It is tied to
  the slate limit, not to the 100 ms step, because a decision over the slate limit puts the slate on air for
  everybody, while a decision over the step only lowers the rate for a moment. With the step as the limit the
  3090 on TensorRT would get a budget of 12 that it does not need (a full batch takes 118 ms, about once after
  a cut), and CUDA fp32 would get K = 2, which leaves most of a crowd under plain covers.
- Deterministic and bounded: a pure function of five measured numbers; the result is 0 or 3..15; it is applied
  once per analyzer start and never changes while the analyzer runs (no controller, nothing to oscillate). A
  slower measurement never gives a larger budget (TESTED on 500 random paths).
- It can only add covers. With a budget, whoever waits for the eye is covered (3.2); without one nothing changes.
- Offline runs never apply it: a benchmark must give the same result on every machine. They still report what
  the rule would choose (`headroom.budget`), so a pod run shows K without a realtime run.
- An explicit `EYE_BUDGET=K` always wins.

Why it is off: the rule has not met a real GPU. Not known yet: how well blank-frame timings at start predict
decisions in a live crowd with OBS encoding on the same card (the 0.6 share is the margin for that, a guess),
and whether Ali accepts the plain covers at the K it picks. E9 measures both. Until then the slow path is loud
but unchanged.

## 4. GPU experiments to run next

`$ONNX` = exported model folder, `$HEADS` = head folder, clips as in run 55. Every offline run writes a diag log;
coverage is checked against the torch run with the same tools as always.

**E0. Reference (once)**
```bash
BACKEND=torch python -m streamsafe --input volleyball.mp4 --output ref.mp4 --mode offline --head-dir $HEADS --stats ref.json --diag ref_diag.json
BACKEND=torch python -m streamsafe --input trailer.mp4 --output ref_t.mp4 --mode offline --head-dir $HEADS --diag ref_t_diag.json
```

**E1. fp16 eye** (lever 1)
```bash
python -m tools.export_onnx $ONNX eye16
EYE_FP16=1 python -m bench.backend_equiv trailer.mp4 $ONNX $HEADS 60        # bar: eye only_torch = 0
for FP in 0 1; do
  BACKEND=onnx ONNX_DIR=$ONNX EYE_FP16=$FP python -m streamsafe --input volleyball.mp4 --output fp$FP.mp4 \
    --mode offline --head-dir $HEADS --stats fp$FP.json --diag fp${FP}_diag.json
done
python -m bench.diag_metrics volleyball.mp4 40 torch=ref_diag.json fp32=fp0_diag.json fp16=fp1_diag.json
python -m bench.disagree volleyball.mp4 torch=ref_diag.json fp16=fp1_diag.json disagree_fp16.jpg   # bar: only_torch = 0, or every cell reviewed
BACKEND=onnx ONNX_DIR=$ONNX EYE_FP16=1 python -m streamsafe --input volleyball.mp4 --output rt_fp16.mp4 \
  --head-dir $HEADS --secs 40 --stats rt_fp16.json          # realtime: decisions_per_s, failclosed_frames
```
Read: `ms_eye`, `ms_total` (mean, p95, max), `embedded` (mean, max), `vram_gb`.

**E2. Per-model speed** (levers 2, 7, 10; sets K for E4)
```bash
for F in eye.onnx eye16.onnx detector.onnx pose.onnx scene_vision.onnx; do
  STREAMSAFE_EP=cuda python -m bench.onnx_speed $ONNX/$F 1,2,4,8,16 30
done
```
Read: `per_item` of eye16 (ms per crop, gives K = eye time allowed / ms per crop); `first_ms` against `p50`
(what the warm-up saves); `switch_extra_ms` (above 1 ms: build lever 7); `bound_p50` against `p50` (lever 10).

**E3. TensorRT** (lever 4). Done in run 57 (numbers in sections 1 and 3.3); kept for the next card (E5).
```bash
pip install "tensorrt-cu12==10.9.*"   # with onnxruntime-gpu 1.22.0 (run 57); no LD_LIBRARY_PATH needed any more: E8 checks that
for F in detector.onnx pose.onnx eye.onnx; do TRT=1 python -m bench.onnx_speed $ONNX/$F 1,2,4,8,16 30; done   # run twice: load_s = build, then load
TRT=1 python -m bench.backend_equiv trailer.mp4 $ONNX $HEADS 60             # bar: eye only_torch = 0, pose kp_px small
TRT=1 BACKEND=onnx ONNX_DIR=$ONNX python -m streamsafe --input volleyball.mp4 --output trt.mp4 --mode offline \
  --head-dir $HEADS --stats trt.json --diag trt_diag.json
python -m bench.disagree volleyball.mp4 torch=ref_diag.json trt=trt_diag.json disagree_trt.jpg
```
If `eye.onnx` does not build: repeat with `TRT=1 EYE_FP16=1`. Note `providers` in the output: `device` must be `trt`.

**E4. Eye budget** (lever 3), with the fastest eye from E1 / E3
```bash
for K in 0 4 6 8; do
  BACKEND=onnx ONNX_DIR=$ONNX EYE_FP16=1 EYE_BUDGET=$K python -m streamsafe --input volleyball.mp4 --output k$K.mp4 \
    --mode offline --head-dir $HEADS --stats k$K.json --diag k${K}_diag.json
done
python -m bench.diag_metrics volleyball.mp4 40 k0=k0_diag.json k4=k4_diag.json k6=k6_diag.json k8=k8_diag.json
python -m bench.disagree volleyball.mp4 k0=k0_diag.json k6=k6_diag.json disagree_k6.jpg
python -m bench.sbs k0.mp4 k6.mp4 sbs_k6.mp4                                # for Ali: the extra plain covers
BACKEND=onnx ONNX_DIR=$ONNX EYE_FP16=1 EYE_BUDGET=6 python -m streamsafe --input volleyball.mp4 --output rt_k6.mp4 \
  --head-dir $HEADS --secs 40 --stats rt_k6.json
```
Bars: `embedded.max` <= K; `ms_eye` and `ms_total` max under 300 ms; realtime `failclosed_frames` = 0 where run 55
had 32; `only_k0` in the disagree line small and every cell reviewed (expected: timing shifts of at most 2
decisions, no person lost); `late_cover` and `torso_under80` not worse. `eye_put_off` (mean, max) says how many
plain covers the budget cost. Repeat on the trailer (cuts are the hard case for the budget).

**E8. TensorRT as a product: cache, fallback, prepare** (section 3.3; about 15 minutes of GPU time)
```bash
pip install "tensorrt-cu12==10.9.*"; unset LD_LIBRARY_PATH        # a: the engine must find the libraries by itself
export TRT=1 TRT_CACHE=/tmp/trt_e8 BACKEND=onnx ONNX_DIR=$ONNX; rm -rf $TRT_CACHE
python -m streamsafe.prepare --models $ONNX --check; echo "exit $?"    # b: exit 3, four "not ready ... (not built yet)"
time python -m streamsafe.prepare --models $ONNX; echo "exit $?"       # c: four "engine built in N s", exit 0; note each N
du -sh $TRT_CACHE/*; ls $TRT_CACHE/*/                                   #    engine sizes; one seal.json per folder, no .building left
time python -m streamsafe.prepare --models $ONNX                        # d: four "verified", a few seconds (the sha256 cost)
python -m streamsafe.syscheck --models $ONNX --engines | grep engines   #    [OK] engines ready (4/4)
TRT_BUILD=0 python -m streamsafe --input volleyball.mp4 --output e8.mp4 --mode offline --head-dir $HEADS \
  --stats e8.json --diag e8_diag.json 2> e8.err                         # e: the engine starts from the prepared engines
python - <<'PY'
import json; s = json.load(open("e8.json"))["summary"]
print(s["gpu"], s["device_notes"], s["headroom"])   # bars: "onnx-trt", [] (a note "the stored engine was not used" = the
PY
#    move into place broke ONNX Runtime's lookup: report it), headroom.budget 0 and holds_rate true on a 3090
python -m bench.diag_metrics volleyball.mp4 40 torch=ref_diag.json trt=e8_diag.json
python -m bench.disagree volleyball.mp4 torch=ref_diag.json trt=e8_diag.json disagree_e8.jpg   # the 97 / 33 pairs of run 57, cell by cell
TRT=1 python -m bench.backend_equiv trailer.mp4 $ONNX $HEADS 60         # eye only_torch = 0 again; read scene p_absdiff and pose kp_px

# f: damage. A cut-off engine must be refused, never loaded
E=$(ls $TRT_CACHE/eye-*/*.engine | head -1); truncate -s -4096 "$E"
python -m streamsafe.prepare --models $ONNX --check                      #    exit 3: "eye ... is damaged (size / sha256 mismatch)"
TRT_BUILD=0 python -m streamsafe --input volleyball.mp4 --output f.mp4 --mode offline --head-dir $HEADS --secs 5 --stats f.json 2> f.err
grep STREAMSAFE_DEVICE f.err                                             #    "onnx-trt+cuda | eye.onnx: TensorRT not used (engine not ready: ...damaged...)"
python -m streamsafe.prepare --models $ONNX                              #    rebuilds the eye only

# g: an interrupted build. Kill prepare in the middle of the eye, then run it again
rm -rf $TRT_CACHE/eye-*; (python -m streamsafe.prepare --models $ONNX & sleep 25; kill -9 $!); ls $TRT_CACHE
python -m streamsafe.prepare --models $ONNX --check                      #    exit 3 for the eye: the .building folder is not an engine
python -m streamsafe.prepare --models $ONNX                              #    exit 0

# h: a driver change, faked through the tool the key reads (a real driver update is the Windows test). Use this pod's card name
mkdir -p /tmp/fakesmi && printf '#!/bin/sh\necho "NVIDIA GeForce RTX 3090, 24576, 999.99"\n' > /tmp/fakesmi/nvidia-smi && chmod +x /tmp/fakesmi/nvidia-smi
PATH=/tmp/fakesmi:$PATH python -m streamsafe.prepare --models $ONNX --check   # exit 3: "the graphics driver changed (<real> -> 999.99)" x4

# i: TensorRT missing. The engine must run on CUDA and say why
pip uninstall -y tensorrt-cu12 tensorrt-cu12-libs tensorrt-cu12-bindings
python -m streamsafe.prepare --models $ONNX; echo "exit $?"              #    exit 2: "PREPARE_FAILED tensorrt: the TensorRT libraries were not found"
python -m streamsafe --input volleyball.mp4 --output i.mp4 --head-dir $HEADS --secs 40 --stats i.json 2> i.err   # realtime on the fallback
grep -E "STREAMSAFE_(DEVICE|SLOW)" i.err                                 #    4 DEVICE lines (onnx-cuda) + one SLOW line; i.json: gpu onnx-cuda,
                                                                         #    decisions_per_s about 5.4 as in run 57, headroom.budget about 7
# i2: TensorRT present by name but broken, so ONNX Runtime itself fails on it (alone it would go on to the CPU)
mkdir -p /tmp/brokentrt && : > /tmp/brokentrt/libnvinfer.so.10
TRT_LIBS=/tmp/brokentrt python -m streamsafe --input volleyball.mp4 --output i2.mp4 --mode offline --head-dir $HEADS --secs 5 --stats i2.json 2> i2.err
grep STREAMSAFE_DEVICE i2.err                                            #    "onnx-cuda | ...: TensorRT not used (the engine could not be built (<ONNX Runtime's error>))"
python -c "import json; print(json.load(open('i2.json'))['summary']['gpu'])"   # onnx-cuda, never onnx-cpu; note the error text ONNX Runtime gave
pip install "tensorrt-cu12==10.9.*"

# j: build at engine start (TRT_BUILD=1, the developer default) with the other models in memory: does it fit, how long?
rm -rf $TRT_CACHE; time python -m streamsafe --input volleyball.mp4 --output j.mp4 --mode offline --head-dir $HEADS --secs 5 --stats j.json
nvidia-smi --query-gpu=memory.used --format=csv                          #    while it builds: peak memory, for the 8 GB question
```
Read: the four build times and engine sizes (they replace the estimates in `prepare.NEED_FREE_GB` and
`docs/NVIDIA_TRT.md`), the time of step d, and every bar above. Any bar that fails is a bug in 3.3, not a tuning matter.

**E9. Automatic eye budget** (section 3.5), on the CUDA fallback, where it matters, and on TensorRT, where it must do nothing
```bash
for AUTO in 0 1; do                                                      # realtime, CUDA fp32 (TRT=0): run 57 had 5.4 decisions/s here
  TRT=0 BACKEND=onnx ONNX_DIR=$ONNX EYE_BUDGET_AUTO=$AUTO python -m streamsafe --input volleyball.mp4 --output auto$AUTO.mp4 \
    --head-dir $HEADS --secs 40 --stats auto$AUTO.json --diag auto${AUTO}_diag.json
done
TRT=1 TRT_BUILD=0 BACKEND=onnx ONNX_DIR=$ONNX EYE_BUDGET_AUTO=1 python -m streamsafe --input volleyball.mp4 --output auto_trt.mp4 \
  --head-dir $HEADS --secs 40 --stats auto_trt.json
K=$(python -c "import json; print(json.load(open('auto1.json'))['summary']['headroom']['budget_applied'])")
TRT=0 BACKEND=onnx ONNX_DIR=$ONNX EYE_BUDGET=$K python -m streamsafe --input volleyball.mp4 --output k_auto.mp4 --mode offline \
  --head-dir $HEADS --stats k_auto.json --diag k_auto_diag.json           # the same K offline, for the coverage tools
python -m bench.diag_metrics volleyball.mp4 40 torch=ref_diag.json k_auto=k_auto_diag.json
python -m bench.sbs auto0.mp4 auto1.mp4 sbs_auto.mp4                     # for Ali: the plain covers the budget costs
```
Read in `summary`: `headroom` (do `fixed_ms`, `eye_one_ms`, `eye_full_ms` match `ms_detect` + `ms_pose` and the
E2 per-crop times? is `decision_ms` for 4 crops near the measured `ms_total` p50?), `budget_applied`,
`decisions_per_s`, `failclosed_frames`, `ms_total` max. Bars: on TensorRT `budget_applied` = 0 and the run equals
E8 e; on CUDA `ms_total` max under 300 ms and fewer `failclosed_frames` with AUTO=1 than with AUTO=0;
`late_cover` / `torso_under80` not worse. Only then, and with Ali's yes to `sbs_auto.mp4`, does it go on by default.

**E5. A consumer-class GPU**: E1, E2 and E4 again on an RTX 3060 / 4060-class card (a rented A4000 is the nearest;
paid, needs Ali's approval). Until then every 3060 number in this file is an ESTIMATE.

**E6. DirectML** (Windows, any DirectX 12 GPU): `STREAMSAFE_EP=dml python -m bench.onnx_speed` for `eye.onnx` and
`eye16.onnx`. Read `switch_extra_ms` (lever 7) and whether fp16 is faster at all on that driver.

**E7. Does a person who did not change keep their score?** `EYE_EVERY=1` offline with `--diag` on three clips, then
for every track compare |p(t) - p(t-1)| with the change of a 16x24 grey thumbnail of the crop. Only if no
change of side (below / above the threshold) hides behind a small thumbnail change is a "look only at changed
people" rule worth building.

## 5. Our own small eye (distillation proposal)

**Step 0, one day, before any distillation.** DOC-CLAIM (model hub, 10 Nov 2026): NVIDIA publishes smaller
siblings of the eye: `nvidia/C-RADIOv3-B`, `nvidia/C-RADIOv3-L`, `nvidia/C-RADIOv4-SO400M`. Embed the labeled
crops with each (`training/build_labeled.py`), train heads (`training/compare_heads.py`), compare on the held-out
families. If B (about 7x fewer operations) or L (about 2x) keeps the misses at zero on Ali's revealing crops, that
is the cheapest win, and B is the best starting point for the student below. Licence per repository to be read
first (the hub lists "other"), and their repository code read before a pin is set, as for the current eye.

**Student.** A ViT-B/16 (86 M parameters, about 7.5x fewer operations per crop than ViT-H) or ViT-S/16 (22 M, about
30x), same input (384x224, same `eye_common.prepare`), plus one linear layer to the teacher's embedding size.
Start weights: C-RADIOv3-B if step 0 shows its licence fits, otherwise SigLIP2-base (Apache-2.0, already shipped
as the scene model, exports cleanly) or DINOv2-S/B (Apache-2.0). The student outputs an embedding in the
teacher's space, so the existing head files, threshold and tracker work unchanged on day one.

**Data** (no labels needed for most of it):
- 300 to 500 thousand person crops cut by the engine itself (`Analyzer.detect`: same masks, same darkened
  background, same full-resolution crops) from all game footage we hold, one crop per track per second,
  near-duplicates removed (cosine > 0.98 in teacher space);
- Open Images person crops through `training/oi_photos.py` (children excluded) for variety;
- every labeled crop and every mined hard crop (`training/mine_hard.py`), weighted 10x;
- no leaked footage, as always.

**Targets.** The teacher's embedding of each crop, computed once with the product torch path (ESTIMATE: 500 k
crops x 6.5 ms = under 1 GPU hour).

**Loss.** `1 - cos(student, teacher)` + smooth L1 on the embedding + the two head logits (fast, male) of the
student against those of the teacher, weighted 4x for crops whose teacher score lies between 0.25x and 4x the
threshold. The heads are fixed directions in embedding space, so this spends the small model's capacity where the
cover decision is made. Augmentation is mild on purpose (flip, small crop jitter, video-compression noise,
slight blur, brightness +-10 %; no hue change: skin against cloth is the signal).

**Training.** AdamW, layer-wise learning-rate decay, cosine schedule, 20 to 30 epochs, fp16, batch 256. ESTIMATE:
4 GPU hours for B, 1.5 for S on an A6000-class card.

**Gates, in this order** (held-out game families as in `training/common.py`):
1. labeled test split, fixed heads, product threshold: no revealing crop that the teacher covers is lost
   (every difference on a contact sheet for Ali); false covers at most 2 points above the teacher;
2. heads re-fit on student embeddings (`compare_heads.py`), threshold set to the teacher's recall, gate 1 again;
3. offline engine runs on trailer, volleyball and club clips: `bench/disagree` teacher against student with no
   track covered only by the teacher after review; `bench/diag_metrics` not worse;
4. `bench/onnx_speed` on a 3060-class card: at most 4 ms per crop (B, fp16) or 2 ms (S).

**How it ships: a two-stage eye.** The student looks at everybody in every decision (no eye skip needed, so
readings are fresher than today). The big eye is called only where the student is unsure: score inside a band
around the threshold, first looks at big people, or student and track history disagreeing. The band is set on
the labeled data so that no revealing crop lies below it. The eye budget of section 3.2 then schedules the big
eye's few looks, and its rule stays: unsure and not yet seen by the big eye means covered. The tracker uses the
student's embedding for identity in every decision. ESTIMATE: 70 to 90 % fewer big-eye calls; to be measured as
the share of crops inside the band.

**Cost.** ESTIMATE: 30 to 60 USD of GPU rent for teacher embeddings, 3 to 4 training rounds and the gates; one to
two weeks of work; one labeling review by Ali (the contact sheets of gate 1). Paid spend needs his approval.

**Risks.** A small model may miss exactly the hard cases the big one was chosen for (dark clubs, small people,
odd poses): that is what gates 1 and 3 and the two-stage design are for. Licence: the student is trained on the
outputs of an NVIDIA Open Model License model; whether that makes it a derivative model, and what follows,
is UNKNOWN until Ali has read the licence text.
