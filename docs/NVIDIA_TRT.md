# NVIDIA edition: TensorRT for the installer and the app

TensorRT is the NVIDIA product path (`TRT=1`). Run 57 on an RTX 3090 measured 74 ms per decision with TensorRT
against 168 ms on CUDA fp32 (volleyball clip), and CUDA fp32 held only 5.4 of the 10 decisions/s in realtime.
This page says what the installer ships, what happens on the first run, and how a new graphics card or driver is
detected. Code: `streamsafe/models/onnx_backend/engines.py`, `session.py`, `streamsafe/prepare.py`.

Claim labels: **TESTED** (run 57, Linux pod, RTX 3090, or the CPU test suite where it says so),
**SOURCE-READ** (read in the named file), **DOC-CLAIM** (vendor documentation), **UNKNOWN** (needs the test
named next to it). Nothing on this page was run on Windows yet.

## 1. What is shipped

| Part | Files | Size on disk | Label |
|---|---|---|---|
| ONNX Runtime GPU 1.22.0 | `onnxruntime.dll` 15 MB, `onnxruntime_providers_cuda.dll` 322 MB, `onnxruntime_providers_tensorrt.dll` 0.8 MB, `onnxruntime_providers_shared.dll` | 0.36 GB | SOURCE-READ: file list of the `onnxruntime_gpu-1.22.0` Windows wheel |
| TensorRT 10.9.0.34 | `nvinfer_10.dll` 441 MB, `nvinfer_plugin_10.dll` 51 MB, `nvonnxparser_10.dll` 3 MB, `nvinfer_builder_resource_10.dll` 1972 MB | 2.47 GB (1.63 GB download) | SOURCE-READ: file list of `tensorrt_cu12_libs-10.9.0.34-…-win_amd64.whl` on pypi.nvidia.com |
| CUDA 12 runtime, cuDNN 9 | `cudart64_12.dll`, `cublas64_12.dll`, `cudnn64_9.dll` and what they load | UNKNOWN: measure on the first Windows install | SOURCE-READ for the names (below) |
| Microsoft Visual C++ runtime | `MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll` | small | SOURCE-READ (below) |

- SOURCE-READ (import table of `onnxruntime_providers_tensorrt.dll`, 1.22.0): it loads `nvinfer_10.dll`,
  `nvinfer_plugin_10.dll`, `nvonnxparser_10.dll`, `cublas64_12.dll`, `cudart64_12.dll`, `cudnn64_9.dll` and the
  Visual C++ runtime. If one is missing, the TensorRT provider does not start and the engine runs on CUDA (section 4).
- The version pair is fixed: ONNX Runtime 1.22.0 with TensorRT 10.9 is what run 57 used (TESTED on Linux,
  `pip install tensorrt-cu12==10.9.*`). Another TensorRT major version has other DLL names and will not load.
- The 2 GB builder resource is needed because engines are built on the streamer's PC. DOC-CLAIM (TensorRT
  documentation): it is used by the builder only, not to run a finished engine.
- So the NVIDIA edition is about 2.5 GB larger than the SPEED_PLAN first guessed ("about 1 GB").
- **Licence, for Ali to read before shipping.** SOURCE-READ (`LICENSE.txt` in the TensorRT wheel, section 12.1):
  the distribution grant names "the libnvinfer and libnvinfer_plugin libraries", in binary form, as part of our
  own application. `nvonnxparser` and the builder resource are not named there. UNKNOWN whether they may be
  shipped inside our installer. The path without that question: the installer downloads NVIDIA's own wheel on the
  streamer's PC (`pip install tensorrt-cu12==10.9.*` into the app's Python), which is also what was tested.

How the engine finds the libraries (SOURCE-READ, UNKNOWN on Windows until tested):
- pip layout: nothing to set. The engine imports NVIDIA's `tensorrt_libs` package, whose own loader loads every
  library in its folder. Run 57 set `LD_LIBRARY_PATH` by hand instead; that still works.
- any other layout: set `TRT_LIBS` to the folder with the four DLLs. The engine adds it to the DLL search path
  and loads the libraries from it.

## 2. The first run: `python -m streamsafe.prepare`

TensorRT compiles every model into an *engine* for this card. TESTED (run 57, RTX 3090): detector 77 s, pose
28 s, eye 58 s; loading the finished engines afterwards takes 6 to 14 s. UNKNOWN: the scene model's build time
(it is the fourth engine), the sizes of the engine files, and all of this on a 3060-class card.

```
python -m streamsafe.prepare --models <model folder> [--cache <engine folder>] [--json]
python -m streamsafe.prepare --models <model folder> --check          # builds nothing, no GPU work
python -m streamsafe.syscheck --models <model folder> --engines       # the same answer as a "check my PC" line
```

It prints one line per event, flushed (with `--json`: one JSON object per line with `event`, `step`, `of`,
`name`, `detail`):

```
verifying: model files
preparing 1/4: scene... (not built yet)
ready 1/4: scene (engine built in 41 s (not built yet))
preparing 2/4: detector... (not built yet)
...
PREPARE_OK 4/4 engines ready
```

| Exit code | Meaning | What the app does |
|---|---|---|
| 0 | every engine is ready | start the engine |
| 2 | failed; the last line is `PREPARE_FAILED <part>: <reason>` | show the reason; the engine can still start and will run that model on CUDA (section 4) |
| 3 | `--check` only: something must be built | run it without `--check` behind a progress bar |

Recommended flow (TESTED with a faked ONNX Runtime, `tests/test_prepare.py`; UNKNOWN on a real GPU):
1. Installer, or the app's first run: `prepare` with a progress bar ("preparing 2/4: eye..."). About 3 minutes.
2. Every app start: `prepare` again. With everything ready it re-hashes the files and builds nothing.
3. Start the engine with `TRT=1 TRT_BUILD=0`: "go live" then never waits for a build. With `TRT_BUILD=1`
   (the default, for developers) an engine that is not ready is built at engine start instead.

Settings: `TRT_CACHE` (engine folder, default `<model folder>/trt_cache`), `TRT_FP16`, `EYE_FP16`, `TRT_LIBS`.
`prepare` and the engine must see the same values, or the engine will not find what was prepared (it then says
"the engine settings changed").

**The engine folder must belong to the streamer's own user.** Use a per-user folder such as
`%LOCALAPPDATA%\StreamSafe\trt_cache`, not `Program Files` (not writable) and not a folder every user can write.
An installer running as another (administrator) account must pass that folder with `--cache`. Reason: engines
are compiled code that the engine loads; the checks below detect damage, not somebody who can rewrite the folder.

`prepare` builds one model at a time and frees it again. A build at engine start (`TRT_BUILD=1`) has the models
loaded before it still in GPU memory: UNKNOWN whether that fits an 8 GB card. One more reason to use `prepare`.

## 3. How a new graphics card or driver is detected

Every engine has a **key**: everything it depends on.

| Part of the key | Read from |
|---|---|
| sha256 of the model file and its weight file | `manifest.json`, already verified |
| name and driver version of every NVIDIA card | `nvidia-smi` (installed with the driver) |
| ONNX Runtime version | `onnxruntime.__version__` |
| TensorRT version | the pip package version, or name + size of `nvinfer_10.dll` in `TRT_LIBS` |
| fp16 flag, batch range, build options | the engine's own settings |

- The engine folder is named after the key (`eye-3f2a…`), and the full key is written into `seal.json` in it.
- At every load the key is computed again and compared with the seal, then every file is checked against the
  size and sha256 in the seal. Only then is TensorRT asked to open the engine.
- So a new card, a driver update, an app update that brings another TensorRT or ONNX Runtime, or new model files
  all lead to "not ready" with the reason in words, for example
  `the graphics driver changed (561.09 -> 566.14)`. The old engine is never opened. `prepare` builds the new
  one and deletes the old folders.
- Expect a rebuild (about 3 minutes, once) after every NVIDIA driver update. The app should say so.
- If `nvidia-smi` cannot be run or the TensorRT libraries are not found, no engine is built or loaded at all
  (an engine for an unknown card could not be told from a stale one). CUDA runs the models.

Damaged and half-built engines:
- A build writes into a private folder (`eye-3f2a….building-<pid>-…`). The seal is written last, after the new
  engine answered a test input with finite numbers, and the folder is then moved into place in one step.
- A build that was interrupted (PC switched off, installer closed) leaves only a `.building` folder, which
  nothing reads. `prepare` deletes such folders after an hour.
- A file cut short, changed, missing or added after the build fails the size / sha256 check and is rebuilt.
- An engine that passes the checks but that TensorRT still refuses is deleted and rebuilt.

TESTED (CPU, faked ONNX Runtime and PC, `tests/test_trt_engines.py`): all of the above. UNKNOWN until the GPU
run in SPEED_PLAN E8: that ONNX Runtime really finds the engine again after the folder was moved, and the cost of
hashing the engine files at every start. On Windows also UNKNOWN: moving a folder with a fresh 1 GB file while a
virus scanner looks at it (the code retries for 5 s, then keeps the engine for this run and builds again next time).

## 4. When TensorRT does not work

The chain per model is **TensorRT, then CUDA, then refuse to start**. The CPU is never used.

| Situation | What runs | Where it shows |
|---|---|---|
| engine ready | TensorRT | `gpu: "onnx-trt"`, `device_notes: []` |
| engine built at this start (`TRT_BUILD=1`) | TensorRT | `device_notes: ["eye.onnx: engine built in 58 s (the graphics driver changed (…))"]` |
| engine not ready and `TRT_BUILD=0`; build failed; libraries missing; engine answers with NaN | CUDA for that model | `gpu: "onnx-cuda"` or `"onnx-trt+cuda"`, `device_notes: ["eye.onnx: TensorRT not used (<why>); running on CUDA"]` |
| CUDA does not start either | nothing: the engine stops with an error | exit, no stream without protection |

- `gpu`, `device_notes` and `headroom` are fields of `summary` in the run report (`--stats`, `LIVE_RESULT`).
- At start the same notes go to stderr, one line each: `STREAMSAFE_DEVICE onnx-trt+cuda | eye.onnx: …`.
- `headroom` holds this PC's model times measured at start and `holds_rate`. When an ordinary decision
  (4 people) does not fit into 100 ms, stderr gets `STREAMSAFE_SLOW this GPU path holds about 8.3 decisions/s …`.
  The app should show both lines to the streamer: the stream is still protected (the gate shows the slate when
  decisions come too far apart), but covers follow less closely. `EYE_BUDGET_AUTO=1` (off by default) then limits
  the eye's work per decision; see SPEED_PLAN.md section 3.5.

SOURCE-READ (ONNX Runtime 1.22.0, `onnxruntime_inference_collection.py`): when TensorRT fails, ONNX Runtime
itself prints the error and retries on the **CPU only**. The engine does not use that session: it opens CUDA
itself and keeps the error text for the report. TESTED with a faked ONNX Runtime; UNKNOWN on a real failure
(E8 forces one).
