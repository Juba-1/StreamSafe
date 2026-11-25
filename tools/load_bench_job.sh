#!/bin/bash
# Load bench (GPU pod, through tools/pod_guard.sh): how long the protection takes to start under each runtime setting.
#   RUN_NAME=load_r1 bash tools/pod_guard.sh tools/load_bench_job.sh
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
        snapshot_download('juba-1/streamsafe-models',local_dir='/workspace/m',allow_patterns=['v1/*'],ignore_patterns=['v1/translate/*'])
        break
    except Exception as e: print('retry', e)
print('DL_OK')
P
mkdir -p /workspace/heads && for f in fast_h6.txt male_h6.txt thr_h6.txt; do curl -sSfL -o /workspace/heads/$f https://raw.githubusercontent.com/Juba-1/StreamSafe-download/main/heads/v3/$f; done; ls -la /workspace/heads
cd $ENG
for t in 1 2; do timeout 900 pip install -q --default-timeout 60 "onnxruntime-gpu>=1.22,<1.23" "tensorrt-cu12-libs>=10.9,<10.10" "tensorrt-cu12-bindings>=10.9,<10.10" opencv-python-headless imageio-ffmpeg 2>&1 | tail -2; python -c "import cv2, onnxruntime" && break; echo "PIP_RETRY $t"; done
pip install -q --no-build-isolation -e . 2>&1 | tail -2; python -c "import streamsafe, cv2, onnxruntime" || { echo PIP_FAIL; exit 1; }
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }
export BACKEND=onnx ONNX_DIR=/workspace/m/v1 TRT_CACHE=/workspace/trt HEAD_DIR=/workspace/heads
TRT=1 python -m streamsafe.prepare --models $ONNX_DIR 2>&1 | grep -v "timing cache" | tail -2
echo "== load bench: each configuration in a fresh process, twice"
B() { local label=$1; shift; for i in 1 2; do env "$@" python tools/load_bench.py "$label#$i" 2>&1 | grep -E "LOADBENCH|Error|error" | tail -2 | tee -a $R/bench.txt; done; }
for c in ${CONFIGS:-trt_prepared cuda}; do
  case $c in
    trt_prepared) B trt_prepared TRT=1 TRT_BUILD=0 ;;
    trt_profile) env TRT=1 TRT_BUILD=0 PROFILE=1 python tools/load_bench.py trt_profile > $R/profile_trt.txt 2>&1; grep LOADBENCH $R/profile_trt.txt | tee -a $R/bench.txt ;;
    cuda) B cuda TRT=0 ;;
    trt_ctx)
      TRT=1 TRT_BUILD=0 python tools/trt_ctx_bench.py record /workspace/ctx 2>&1 | grep -E "RECORDED|Error|error" | tail -3 | tee -a $R/bench.txt
      for mode in normal ctx normal_parallel ctx_parallel normal ctx; do
        TRT=1 python tools/trt_ctx_bench.py time /workspace/ctx $mode 2>&1 | grep -E "CTXBENCH|Error|error" | tail -2 | tee -a $R/bench.txt
      done ;;
    cuda_heuristic) B cuda_heuristic TRT=0 CUDNN_SEARCH=HEURISTIC ;;
    trt_not_prepared) B trt_not_prepared TRT=1 TRT_BUILD=0 TRT_CACHE=/workspace/trt_empty ;;
  esac
done
echo "== DONE $(date -u)"
