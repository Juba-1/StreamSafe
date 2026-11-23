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
T=os.environ['HF_TOKEN']; R='juba-1/streamsafe-work'
for i in range(5):
    try:
        hf_hub_download(R,'data/gta6_v1.tar',repo_type='dataset',token=T,local_dir='/workspace/vt')
        snapshot_download(R,repo_type='dataset',token=T,local_dir='/workspace/h',allow_patterns=['runs/run48_lucia_runner/heads_lucia/ali/*.txt'])
        snapshot_download('juba-1/streamsafe-models',local_dir='/workspace/m',allow_patterns=['v1/*'],ignore_patterns=['v1/translate/*'])
        break
    except Exception as e: print('retry', e)
print('DL_OK')
P
mkdir -p v && tar xf vt/data/gta6_v1.tar -C v && rm -rf vt && ls v/gta6 | wc -l
cd $ENG
for t in 1 2; do timeout 900 pip install -q --default-timeout 60 "onnxruntime-gpu>=1.22,<1.23" "tensorrt-cu12-libs>=10.9,<10.10" "tensorrt-cu12-bindings>=10.9,<10.10" opencv-python-headless imageio-ffmpeg 2>&1 | tail -2; python -c "import cv2, onnxruntime" && break; echo "PIP_RETRY $t"; done
pip install -q --no-build-isolation -e . 2>&1 | tail -2; python -c "import streamsafe, cv2, onnxruntime" || { echo PIP_FAIL; exit 1; }
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }
ffmpeg -hide_banner -protocols 2>/dev/null | grep -E "^\s*(srt|rtmps)\s*$" | tr '\n' ' '; echo "<- protocols"
export BACKEND=onnx ONNX_DIR=/workspace/m/v1 TRT_CACHE=/workspace/trt HEAD_DIR=/workspace/h/runs/run48_lucia_runner/heads_lucia/ali
TRT=1 python -m streamsafe.prepare --models $ONNX_DIR 2>&1 | grep -v "timing cache" | tail -3
export TRT=1 TRT_BUILD=0 LOGO=$ENG/marketing/logo_rgba.png SCENE_IMAGE=$ENG/marketing/logo.png SLATE_IMAGE=$ENG/marketing/logo.png SCENE_TEXT= SLATE_TEXT=


echo "== sensitivity"
for c in beach_01 beach_14 club_05 nightclub_08 stripclub_10 npc_19 street_28 night_25; do
  f=/workspace/v/gta6/$c.mp4; [ -f "$f" ] || continue
  ffmpeg -hide_banner -loglevel error -ss 30 -i "$f" -t $SECS -an -c:v libx264 -preset veryfast -crf 20 -y /workspace/in_$c.mp4
  for lv in light:2.0 mid:1.4 high:0.8; do
    n=${lv%%:*}; k=${lv##*:}
    THR_SCALE=$k timeout 900 python -m streamsafe.cli --input /workspace/in_$c.mp4 --output /workspace/o_${c}_$n.mp4 \
      --mode offline --size 1280x720 --fps 15 --stats $R/stats_${c}_$n.json --no-report > $R/eng_${c}_$n.txt 2>&1 || echo "ENG_FAIL $c $n"
  done
  # sheet: rows = 6 times, columns = light | balanced | high
  rows=""
  for t in 5 15 25 35 45 55; do
    for n in light mid high; do ffmpeg -hide_banner -loglevel error -ss $t -i /workspace/o_${c}_$n.mp4 -frames:v 1 -vf scale=480:270 -y /workspace/f_${n}_$t.png; done
    ffmpeg -hide_banner -loglevel error -i /workspace/f_light_$t.png -i /workspace/f_mid_$t.png -i /workspace/f_high_$t.png -filter_complex hstack=3 -y /workspace/row_$t.png
    rows="$rows -i /workspace/row_$t.png"
  done
  ffmpeg -hide_banner -loglevel error $rows -filter_complex vstack=6 -q:v 4 -y $R/sheet_$c.jpg && echo "SHEET $c"
done
python - "$R" <<'P'
import json, pathlib, sys
R = pathlib.Path(sys.argv[1]); out = []
for f in sorted(R.glob("stats_*.json")):
    s = json.loads(f.read_text()); c, n = f.stem[6:].rsplit("_", 1)
    out.append(f"{c:14s} {n:6s} " + json.dumps({k: v for k, v in s.items() if not isinstance(v, (list, dict))}))
(R / "summary.txt").write_text("\n".join(out) + "\n"); print("\n".join(out))
P
echo "== DONE $(date -u)"
