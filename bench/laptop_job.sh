#!/bin/bash
# GPU job: the engine on a laptop-class NVIDIA card, with and without a game-like load on the same card.
#   RUN_NAME=perf_a4000_r1 [PLAN="short crowd long offline lag"] [BASE_SHA=<main commit>] bash bench/laptop_job.sh
# Started on the pod through tools/pod_guard.sh from the staged code tarball. Realtime runs of the GTA 6 trailer,
# a crowd clip and a 10-minute mix, on TensorRT and on plain CUDA (the stand-in for DirectML and for an NVIDIA card
# without the fast mode), each with bench/gpu_load.py at several levels. "old" runs use the code of main
# (BASE_SHA, fetched from the work store's history); "new" runs use this tree. Results: runs/$RUN_NAME on the
# work store (stats, logs, tables; no video).
set -u
RUN_NAME=${RUN_NAME:?}; PLAN=${PLAN:-short crowd long offline lag}
ENG=$(cd "$(dirname "$0")/.." && pwd)
R=/workspace/res; mkdir -p $R /workspace/d; cd /workspace
exec > >(tee -a $R/log.txt) 2>&1
up() { python - <<P
import os
from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_folder(folder_path='$R', path_in_repo='runs/$RUN_NAME', repo_id='juba-1/streamsafe-work', repo_type='dataset', commit_message='$RUN_NAME', ignore_patterns=['*_diag.json'])
print('UPLOADED', flush=True)
P
}
pip install -q huggingface_hub hf_transfer 2>&1 | tail -1
trap up EXIT
echo "== START $(date -u) $RUN_NAME plan: $PLAN"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; free -g | sed -n 2p
BASE_SHA=${BASE_SHA:-} python - <<'P' || { echo DL_FAIL; exit 1; }
import os
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
T=os.environ['HF_TOKEN']; R='juba-1/streamsafe-work'
for i in range(5):
    try:
        hf_hub_download(R,'streamsafe_videos_v1.tar',repo_type='dataset',token=T,local_dir='/workspace/vt')
        snapshot_download(R,repo_type='dataset',token=T,local_dir='/workspace/h',allow_patterns=['runs/run48_lucia_runner/heads_lucia/ali/*.txt'])
        snapshot_download('juba-1/streamsafe-models',local_dir='/workspace/m',allow_patterns=['v1/*'],ignore_patterns=['v1/translate/*'])
        break
    except Exception as e: print('retry', e)
print('DL_OK')
base=os.environ.get('BASE_SHA','')
if base:
    try:  # the code of main as it was staged before this branch: the "old" side of every comparison
        hit=[c for c in HfApi(token=T).list_repo_commits(R,repo_type='dataset') if c.title.strip()==f'stage code {base}']
        hf_hub_download(R,'code/engine3_main.tar.gz',repo_type='dataset',token=T,revision=hit[0].commit_id,local_dir='/workspace/basec')
        print('BASE_OK',hit[0].commit_id)
    except Exception as e: print('BASE_MISSING',repr(e)[:200])
P
mkdir -p v && tar xf vt/streamsafe_videos_v1.tar -C v && rm -rf vt
find /workspace/v -name '*.mp4' | sort | tee $R/clips.txt
BASE=""
if [ -f basec/code/engine3_main.tar.gz ]; then
  mkdir -p base && tar xzf basec/code/engine3_main.tar.gz -C base && BASE=/workspace/base
  # reporting only: main lists at most 200 slate frames, too few to count the episodes of a bad run
  sed -i 's/^MAX_LISTED = 200$/MAX_LISTED = 200000/' base/streamsafe/runtime/report.py
fi
echo "BASE=${BASE:-none (old runs = this tree with the new switches off)}"
cd $ENG
for t in 1 2; do timeout 900 pip install -q --default-timeout 60 "onnxruntime-gpu>=1.22,<1.23" "tensorrt-cu12-libs>=10.9,<10.10" "tensorrt-cu12-bindings>=10.9,<10.10" opencv-python-headless imageio-ffmpeg 2>&1 | tail -2; python -c "import cv2, onnxruntime" && break; echo "PIP_RETRY $t"; done
pip install -q --no-build-isolation -e . 2>&1 | tail -2; python -c "import streamsafe, cv2, onnxruntime" || { echo PIP_FAIL; exit 1; }
command -v ffmpeg || ln -sf "$(python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')" /usr/local/bin/ffmpeg
export BACKEND=onnx ONNX_DIR=/workspace/m/v1 TRT_CACHE=/workspace/trt HEAD_DIR=/workspace/h/runs/run48_lucia_runner/heads_lucia/ali
TRT=1 python -m streamsafe.prepare --models $ONNX_DIR 2>&1 | grep -v "timing cache" | tail -6
T1=$(grep -i '/gta6t1_' $R/clips.txt | head -1)
CROWD=$(grep -i -E 'yakuza_120|volley' $R/clips.txt | head -1)
echo "T1=$T1 CROWD=$CROWD"

# ---- how busy does each load level really keep the card (alone)?
for L in 40 70 90; do
  python -m bench.gpu_load --duty 0.$L --secs 9 > /dev/null & sleep 3
  U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 & P=$!; sleep 5; kill $P)
  wait; echo "LOAD_ALONE $L% duty -> utilization samples: $(echo $U | tr '\n' ' ')"
done

# run NAME CLIP SECS trt|cuda old|new LOAD [late] [VAR=VALUE ...]
run() {
  local name=$1 clip=$2 secs=$3 path=$4 ver=$5 load=$6; shift 6
  local after=0; [ "${1:-}" = late ] && { after=45; shift; }
  [ -n "$clip" ] || { echo "NO_CLIP $name"; return; }
  local dir=$ENG vars=()
  [ $path = trt ] && vars+=(TRT=1 TRT_BUILD=0) || vars+=(TRT=0)
  if [ $ver = old ]; then
    if [ -n "$BASE" ]; then dir=$BASE; else vars+=(EYE_BUDGET_AUTO=0 EYE_BUDGET_ADAPT=0 GARMENT_THREADS=1); fi
  fi
  echo "== RUN $name ($path $ver load $load after $after s, $secs s) $(date -u +%T)"
  local lp=""
  if [ "$load" != 0 ]; then python -m bench.gpu_load --duty 0.$load --after $after > $R/${name}_load.log 2>&1 & lp=$!; sleep 4; fi
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits -l 1 > $R/${name}_util.csv & local sp=$!
  (cd $dir && env -u HF_TOKEN "${vars[@]}" "$@" timeout $((secs + 600)) python -u -m streamsafe --input "$clip" --mode realtime --size 1920x1080 \
     --secs $secs --diag /workspace/d/${name}_diag.json --stats $R/${name}_st.json --no-report > $R/${name}_run.log 2>&1); echo "RUN_EXIT $?"
  kill $sp $lp 2>/dev/null; wait $sp $lp 2>/dev/null
  grep -E "STREAMSAFE_(SLOW|DEVICE)|Traceback|Error" $R/${name}_run.log | head -5
  ln -sf /workspace/d/${name}_diag.json $R/${name}_diag.json
  python -m bench.laptop_summ $R $name | tail -1
}
has() { case " $PLAN " in *" $1 "*) return 0;; esac; return 1; }

if has short; then  # the field report: Trailer 1, 91 s, a cut every 2-3 s
  run t1_trt_old_L0   "$T1" 91 trt  old 0
  run t1_cuda_old_L0  "$T1" 91 cuda old 0
  run t1_trt_old_L70  "$T1" 91 trt  old 70
  run t1_cuda_old_L70 "$T1" 91 cuda old 70
  for L in 0 40 70 90; do
    run t1_trt_new_L$L  "$T1" 91 trt  new $L
    run t1_cuda_new_L$L "$T1" 91 cuda new $L
  done
  run t1_trt_new_L70late  "$T1" 91 trt  new 70 late   # the game starts 45 s into the stream: the start-up measurement saw none of it
  run t1_cuda_new_L70late "$T1" 91 cuda new 70 late
  up
fi
if has crowd; then  # 8 to 15 people in view all the time
  run cr_cuda_old_L0  "$CROWD" 60 cuda old 0
  run cr_cuda_old_L70 "$CROWD" 60 cuda old 70
  run cr_trt_old_L70  "$CROWD" 60 trt  old 70
  run cr_cuda_new_L0  "$CROWD" 60 cuda new 0
  run cr_cuda_new_L70 "$CROWD" 60 cuda new 70
  run cr_trt_new_L70  "$CROWD" 60 trt  new 70
  run cr_trt_new_L90  "$CROWD" 60 trt  new 90
  run cr_cuda_new_L90 "$CROWD" 60 cuda new 90
  up
fi
if has long; then  # 10.5 minutes of every clip we hold, one after the other, under a heavy game
  mkdir -p /workspace/parts; : > /workspace/parts/list.txt; n=0
  while read -r f; do
    n=$((n + 1)); ffmpeg -nostdin -loglevel error -y -t 100 -i "$f" -an -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1,fps=30" \
      -c:v libx264 -preset ultrafast -crf 18 /workspace/parts/p$n.mp4 && echo "file '/workspace/parts/p$n.mp4'" >> /workspace/parts/list.txt
  done < $R/clips.txt
  cat /workspace/parts/list.txt /workspace/parts/list.txt /workspace/parts/list.txt > /workspace/parts/list3.txt
  ffmpeg -nostdin -loglevel error -y -f concat -safe 0 -i /workspace/parts/list3.txt -t 640 -c copy /workspace/long.mp4; ls -la /workspace/long.mp4
  run long_trt_new_L70  /workspace/long.mp4 630 trt  new 70
  run long_cuda_new_L70 /workspace/long.mp4 630 cuda new 70
  run long_cuda_old_L70 /workspace/long.mp4 630 cuda old 70
  up
fi
if has offline; then  # the same decisions on every machine: what do fewer looks cost in cover, and what do garment threads save?
  for cfg in "k0 EYE_BUDGET=0 GARMENT_THREADS=1" "k0t EYE_BUDGET=0" "k3 EYE_BUDGET=3" "k1 EYE_BUDGET=1"; do
    set -- $cfg; n=$1; shift
    env -u HF_TOKEN TRT=1 TRT_BUILD=0 "$@" python -u -m streamsafe --input "$T1" --mode offline --size 1920x1080 --diag /workspace/d/off_${n}_diag.json \
      --stats $R/off_${n}_st.json --no-report > $R/off_${n}_run.log 2>&1; echo "OFFLINE $n exit $?"
    python - $R/off_${n}_st.json <<'P'
import json,sys
r=json.load(open(sys.argv[1]))
print('OFF', sys.argv[1].split('/')[-1], {k:(r[k]['mean'],r[k]['p95'],r[k]['max']) for k in ('ms_total','ms_eye','ms_garment','ms_pose','embedded','eye_put_off','risky') if k in r})
P
  done
  cmp /workspace/d/off_k0_diag.json /workspace/d/off_k0t_diag.json && echo "THREADS_SAME_DIAG"
  up
fi
if has lag; then  # garment against an independent per-frame detection of the same frames (bench/diag_metrics)
  timeout 600 pip install -q --default-timeout 60 rfdetr 2>&1 | tail -1
  D=/workspace/d; L=""
  for n in t1_trt_old_L0 t1_cuda_old_L0 t1_cuda_old_L70 t1_trt_new_L0 t1_cuda_new_L0 t1_trt_new_L70 t1_cuda_new_L70 t1_cuda_new_L90; do [ -f $D/${n}_diag.json ] && L="$L $n=$D/${n}_diag.json"; done
  for n in k0 k3 k1; do [ -f $D/off_${n}_diag.json ] && L="$L off_$n=$D/off_${n}_diag.json"; done
  [ -n "$L" ] && timeout 1500 python -m bench.diag_metrics "$T1" 90 $L > $R/lag_metrics.txt 2>&1; tail -20 $R/lag_metrics.txt | cut -c1-900
fi
echo "== TABLE"; python -m bench.laptop_summ $R | tee $R/table.md
echo "== DONE $(date -u)"
