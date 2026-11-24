#!/usr/bin/env bash
# RunPod job: a short real stream to Kick driven only through the app's window (tools/ui_live_drive.py).
# Env: RUN_NAME, HF_TOKEN, STREAM_URL (secret), TOUR_S (default 240).
set -u
RUN_NAME=${RUN_NAME:?}; : "${STREAM_URL:?STREAM_URL is not set}"
ENG=$(cd "$(dirname "$0")/.." && pwd)
R=/workspace/res; mkdir -p $R; cd /workspace
exec > >(tee -a $R/log.txt) 2>&1
scrub() {
  python3 - "$R" <<'P'
import os, pathlib, sys
u = os.environ["STREAM_URL"].strip(); k = u.rstrip("/").rsplit("/", 1)[-1]
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
from huggingface_hub import hf_hub_download
for i in range(5):
    try:
        hf_hub_download('juba-1/streamsafe-work','streamsafe_videos_v1.tar',repo_type='dataset',token=os.environ['HF_TOKEN'],local_dir='/workspace/vt'); break
    except Exception as e: print('retry', e)
print('DL_OK')
P
mkdir -p v && tar xf vt/streamsafe_videos_v1.tar -C v && rm -rf vt
export TRAILER=$(find /workspace/v -name 'gta6t1_*.mp4' | head -1); echo "TRAILER=$TRAILER"
cd $ENG
for t in 1 2; do timeout 900 pip install -q --default-timeout 60 "onnxruntime-gpu>=1.22,<1.23" "tensorrt-cu12-libs>=10.9,<10.10" "tensorrt-cu12-bindings>=10.9,<10.10" opencv-python-headless imageio-ffmpeg "playwright>=1.47" 2>&1 | tail -2; python -c "import cv2, onnxruntime, playwright" && break; echo "PIP_RETRY $t"; done
pip install -q --no-build-isolation -e . 2>&1 | tail -2; python -c "import streamsafe" || { echo PIP_FAIL; exit 1; }
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }
python -m playwright install --with-deps chromium 2>&1 | tail -1
command -v Xvfb >/dev/null || { apt-get update -qq && apt-get install -y -qq xvfb >/dev/null; }
Xvfb :99 -screen 0 1600x900x24 -nolisten tcp >/dev/null 2>&1 &
export DISPLAY=:99
export STREAMSAFE_DATA_DIR=/workspace/ssdata OUT=$R
# models + heads through the app first (no window yet), then TensorRT engines like "fast mode" does
python - <<'P'
import time
from streamsafe.app.webui import server
a = server.WebApp()
if a.ctrl.models_problem():
    a.do_models({})
    while a.job_name: time.sleep(5)
    print("MODELS:", a.message[0] or "ok", a.ctrl.models_problem() or "installed", flush=True)
P
MD=$(python -c "from streamsafe.app import paths; print(paths.models_dir())"); TC=$(python -c "from streamsafe.app import paths; print(paths.trt_cache_dir())")
TRT=1 TRT_CACHE=$TC python -m streamsafe.prepare --models $MD 2>&1 | grep -v "timing cache" | tail -3
python -u tools/ui_live_drive.py; echo "DRIVE_EXIT $?"
for f in $(ls -t $STREAMSAFE_DATA_DIR/reports/session_*.txt 2>/dev/null | head -1); do cp $f $R/; done
cp $STREAMSAFE_DATA_DIR/webui.log $R/ 2>/dev/null
echo "== END $(date -u)"
