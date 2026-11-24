#!/bin/bash
# Sensitivity check on new GTA 6 clips (GPU pod, through tools/pod_guard.sh): the real engine, offline, at the
# three slider levels (light 2.0 / balanced 1.4 / high 0.8); per clip one sheet (rows = times, columns = levels)
# and the number of covered person-decisions per level.
#   RUN_NAME=sens_r1 bash tools/sens_eval_job.sh
# (header below shared with) Demo live on a REAL platform (GPU pod, through tools/pod_guard.sh): the GTA 6 trailer as the game -> the engine
# with the real models -> a local relay that sends to the platform ($STREAM_URL, a RunPod secret, never printed)
# AND keeps what viewers get (viewer_view.mkv) for the tutorial video.
#   RUN_NAME=demo_live_r1 [MINUTES=6] bash tools/demo_live_job.sh
set -u
RUN_NAME=${RUN_NAME:?}; SECS=${SECS:-60}
ENG=$(cd "$(dirname "$0")/.." && pwd)
R=/workspace/res; mkdir -p $R; cd /workspace
exec > >(tee -a $R/log.txt) 2>&1
scrub() {  # every file: the address and the key out
  python3 - "$R" <<'P'
import os, pathlib, sys
u = os.environ.get("STREAM_URL", "").strip() or "\x00none"; k = u.rstrip("/").rsplit("/", 1)[-1] or "\x00none"
for p in pathlib.Path(sys.argv[1]).rglob("*"):
    if p.is_file() and p.suffix in (".txt", ".log", ".json", ".md"):
        t = p.read_text(encoding="utf-8", errors="replace")
        if u in t or k in t:
            p.write_text(t.replace(u, "***").replace(k, "***"), encoding="utf-8")
P
}
up() { scrub; python - <<P
import os
from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_folder(folder_path='$R', path_in_repo='runs/$RUN_NAME', repo_id='juba-1/streamsafe-work', repo_type='dataset', commit_message='$RUN_NAME')
print('UPLOADED', flush=True)
P
}
pip install -q huggingface_hub hf_transfer 2>&1 | tail -1
trap up EXIT
echo "== START $(date -u) $RUN_NAME"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'P' || { echo DL_FAIL; exit 1; }
import os
from huggingface_hub import hf_hub_download, snapshot_download
T=os.environ['HF_TOKEN']
for i in range(5):
    try:
        hf_hub_download('juba-1/streamsafe-work','streamsafe_videos_v1.tar',repo_type='dataset',token=T,local_dir='/workspace/vt')
        snapshot_download('juba-1/streamsafe-models',local_dir='/workspace/m',allow_patterns=['v1/*'],ignore_patterns=['v1/translate/*'])
        break
    except Exception as e: print('retry', e)
print('DL_OK')
P
mkdir -p v && tar xf vt/streamsafe_videos_v1.tar -C v && rm -rf vt
TR=$(find /workspace/v -name 'gta6t1_*.mp4' | head -1); echo "TRAILER=$TR"
mkdir -p /workspace/heads && for f in fast_h6.txt male_h6.txt thr_h6.txt; do curl -sSfL -o /workspace/heads/$f https://raw.githubusercontent.com/Juba-1/StreamSafe-download/main/heads/v3/$f; done; ls -la /workspace/heads
cd $ENG
for t in 1 2; do timeout 900 pip install -q --default-timeout 60 "onnxruntime-gpu>=1.22,<1.23" "tensorrt-cu12-libs>=10.9,<10.10" "tensorrt-cu12-bindings>=10.9,<10.10" opencv-python-headless imageio-ffmpeg 2>&1 | tail -2; python -c "import cv2, onnxruntime" && break; echo "PIP_RETRY $t"; done
pip install -q --no-build-isolation -e . 2>&1 | tail -2; python -c "import streamsafe, cv2, onnxruntime" || { echo PIP_FAIL; exit 1; }
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }
export BACKEND=onnx ONNX_DIR=/workspace/m/v1 TRT_CACHE=/workspace/trt HEAD_DIR=/workspace/heads
TRT=1 python -m streamsafe.prepare --models $ONNX_DIR 2>&1 | grep -v "timing cache" | tail -2
export TRT=1 TRT_BUILD=0 LOGO=$ENG/marketing/logo_rgba.png SCENE_IMAGE=$ENG/marketing/logo.png SLATE_IMAGE=$ENG/marketing/logo.png SCENE_TEXT= SLATE_TEXT=
echo "== trailer at each sensitivity (realtime-like policy, offline speed)"
# each run: <mode>-<fps>-<k>; realtime = what a stream does (first-look covers, the speed governor)
for run in ${RUNS:-offline-30-1.0 realtime-30-1.0 realtime-30-1.3 realtime-30-1.6 realtime-60-1.0}; do
  IFS=- read mode fps k <<< "$run"
  THR_SCALE=$k timeout 1200 python -m streamsafe.cli --input "$TR" --output /workspace/o_$run.mp4 --mode $mode --size 1920x1080 --fps $fps \
    --delay 2 --stats $R/stats_$run.json --no-report > $R/eng_$run.txt 2>&1 || echo "ENG_FAIL $run"
done
for run in ${RUNS:-offline-30-1.0 realtime-30-1.0 realtime-30-1.3 realtime-30-1.6 realtime-60-1.0}; do  # every 2 s, like the window's preview
  ffmpeg -hide_banner -loglevel error -i /workspace/o_$run.mp4 -vf "fps=0.5,scale=320:180,tile=6x8" -frames:v 1 -q:v 4 -y $R/sheet_$run.jpg && echo "SHEET $run"
done
python - "$R" <<'P'
import json, pathlib, sys
R = pathlib.Path(sys.argv[1]); out = []
for f in sorted(R.glob("stats_*.json")):
    s = json.loads(f.read_text()); sm = s.get("summary", s)
    keep = {k: v for k, v in sm.items() if not isinstance(v, (list, dict))}
    for k in ("plain_decisions", "risky_decisions", "first_look", "plain_covers", "covered_people", "people"):
        v = s.get(k)
        if isinstance(v, list): keep[k] = len(v) if v and not isinstance(v[0], (int, float)) else sum(v)
    keep["list_keys"] = sorted(k for k, v in s.items() if isinstance(v, list))[:60]
    out.append(f.stem + " " + json.dumps(keep))
(R / "summary.txt").write_text("\n".join(out) + "\n"); print("\n".join(out))
P
echo "== DONE $(date -u)"
