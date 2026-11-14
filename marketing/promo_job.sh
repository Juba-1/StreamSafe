#!/bin/bash
set -u
mkdir -p /workspace/res && cd /workspace
exec > >(tee -a /workspace/res/log.txt) 2>&1
up() { python - <<'P'
import os
from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_folder(folder_path='/workspace/res', path_in_repo='runs/promo4', repo_id='juba-1/streamsafe-work', repo_type='dataset', commit_message='promo4', ignore_patterns=['raw/*'])
print('UPLOADED', flush=True)
P
}
pip install -q huggingface_hub hf_transfer 2>&1 | tail -1
trap up EXIT
echo "== START $(date -u)"; nvidia-smi --query-gpu=name --format=csv,noheader
python - <<'P' || { echo DL_FAIL; exit 1; }
import os
from huggingface_hub import hf_hub_download, snapshot_download
T=os.environ['HF_TOKEN']; R='juba-1/streamsafe-work'
for i in range(5):
    try:
        for f,d in (('code/engine3_main.tar.gz','/workspace/c'),('streamsafe_videos_v1.tar','/workspace/vt')):
            hf_hub_download(R,f,repo_type='dataset',token=T,local_dir=d)
        snapshot_download(R,repo_type='dataset',token=T,local_dir='/workspace/h',allow_patterns=['runs/run48_lucia_runner/heads_lucia/ali/*.txt'])
        snapshot_download('juba-1/streamsafe-models',local_dir='/workspace/m',allow_patterns=['v1/*'])
        break
    except Exception as e: print('retry', e)
print('DL_OK')
P
mkdir -p eng3 v && tar xzf c/code/engine3_main.tar.gz -C eng3 && tar xf vt/streamsafe_videos_v1.tar -C v && rm -rf vt
find /workspace/v -name '*.mp4' | grep -i gta6 | tee res/clips.txt
cd eng3 && for t in 1 2; do timeout 600 pip install -q --default-timeout 60 "onnxruntime-gpu>=1.22,<1.23" "tensorrt-cu12-libs>=10.9,<10.10" "tensorrt-cu12-bindings>=10.9,<10.10" opencv-python-headless imageio-ffmpeg 2>&1 | tail -2 && python -c "import cv2, onnxruntime, tensorrt" && break; echo "PIP_RETRY $t"; done
pip install -q --no-build-isolation -e '.[translate]' 2>&1 | tail -2; python -c "import streamsafe, cv2, onnxruntime" || { echo PIP_FAIL; exit 1; }
command -v ffmpeg || ln -sf "$(python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')" /usr/local/bin/ffmpeg
export BACKEND=onnx ONNX_DIR=/workspace/m/v1 TRT=1 TRT_CACHE=/workspace/trt HEAD_DIR=/workspace/h/runs/run48_lucia_runner/heads_lucia/ali
L=$PWD/marketing/logo.png
python -m streamsafe.prepare --models $ONNX_DIR 2>&1 | tail -2
R=/workspace/res; mkdir -p $R/raw
for c in gta6t1_; do
  IN=$(grep "/$c" $R/clips.txt | head -1); [ -n "$IN" ] || { echo "NO_CLIP $c"; continue; }
  n=${c%_}; echo "== $n $IN $(date -u)"
  env -u HF_TOKEN TRANSLATE_AR=1 TRANSLATE_DIR=/workspace/m/v1/translate LOGO=$PWD/marketing/logo_rgba.png SCENE_IMAGE=$L SLATE_IMAGE=$L SCENE_TEXT= SLATE_TEXT= timeout 2400 python -u -m streamsafe --input "$IN" --mode realtime --size 1920x1080 \
     --output $R/raw/${n}_prot.mp4 --diag $R/${n}_diag.json --stats $R/${n}_st.json --no-report > $R/${n}_run.log 2>&1; echo "RUN_EXIT $?"; tail -3 $R/${n}_run.log
  bash marketing/make_promo.sh $R/raw/${n}_prot.mp4 $R/${n}_promo.mp4 && echo "PROMO_OK $n"
  python - $R/raw/${n}_prot.mp4 $R/sheets_$n <<'P'
import cv2, sys, os, numpy as np
cap=cv2.VideoCapture(sys.argv[1]); os.makedirs(sys.argv[2],exist_ok=True); tiles=[]; i=0; k=0
def flush():
    global tiles,k
    if not tiles: return
    while len(tiles)<30: tiles.append(np.zeros_like(tiles[0]))
    rows=[np.hstack(tiles[r*6:(r+1)*6]) for r in range(5)]
    cv2.imwrite(f'{sys.argv[2]}/s{k:02d}.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,82]); k+=1; tiles=[]
while True:
    ok,f=cap.read()
    if not ok: break
    if i%15==0:
        t=cv2.resize(f,(320,180),interpolation=cv2.INTER_AREA)
        cv2.putText(t,str(i),(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        tiles.append(t)
        if len(tiles)==30: flush()
    i+=1
flush(); print('SHEETS',k,'frames',i)
P
  if [ "$n" = gta6t1 ]; then mkdir -p $R/frames_$n; for f in 810 870 885 1035 1170 1185 1725 1740 2070 2085 2100; do
    ffmpeg -nostdin -loglevel error -y -i $R/raw/${n}_prot.mp4 -vf "select=eq(n\,$f)" -frames:v 1 -q:v 3 $R/frames_$n/f$f.jpg; done; fi
done
echo "== DONE $(date -u)"
