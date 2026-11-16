#!/bin/bash
# Live end-to-end on a REAL platform, run on a GPU pod:
#   OBS stand-in (FFmpeg, the GTA 6 trailer looped, over SRT) -> the engine with the real models -> RTMPS ($STREAM_URL)
#   RUN_NAME=live_kick_r1 [MINUTES=8] bash tools/live_e2e_job.sh      (through tools/pod_guard.sh)
# Faults on purpose while live: OBS stops and comes back (at OBS_OFF_S), the connection to the platform is cut
# (at NET_CUT_S). A background job of a script starts with Ctrl+C ignored (POSIX), so the engine gets the
# normal Ctrl+C handler back explicitly: the stop at the end then works like the Stop button (report written). Before the main run: a raw push to the address form many users paste (Kick without /app) and the
# app's "Test connection" on the right one. STREAM_URL comes from a RunPod secret; it is never printed, and every
# file is scrubbed of it before upload to the work store (runs/$RUN_NAME).
set -u
RUN_NAME=${RUN_NAME:?}; MINUTES=${MINUTES:-8}; OBS_OFF_S=${OBS_OFF_S:-150}; NET_CUT_S=${NET_CUT_S:-260}
: "${STREAM_URL:?STREAM_URL is not set}"
ENG=$(cd "$(dirname "$0")/.." && pwd)
R=/workspace/res; mkdir -p $R; cd /workspace
exec > >(tee -a $R/log.txt) 2>&1
scrub() {  # every file: the address and the key out
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
from huggingface_hub import hf_hub_download, snapshot_download
T=os.environ['HF_TOKEN']; R='juba-1/streamsafe-work'
for i in range(5):
    try:
        hf_hub_download(R,'streamsafe_videos_v1.tar',repo_type='dataset',token=T,local_dir='/workspace/vt')
        snapshot_download(R,repo_type='dataset',token=T,local_dir='/workspace/h',allow_patterns=['runs/run48_lucia_runner/heads_lucia/ali/*.txt'])
        snapshot_download('juba-1/streamsafe-models',local_dir='/workspace/m',allow_patterns=['v1/*'],ignore_patterns=['v1/translate/*'])
        break
    except Exception as e: print('retry', e)
print('DL_OK')
P
mkdir -p v && tar xf vt/streamsafe_videos_v1.tar -C v && rm -rf vt
T1=$(find /workspace/v -name 'gta6t1_*.mp4' | head -1); echo "T1=$T1"
cd $ENG
for t in 1 2; do timeout 900 pip install -q --default-timeout 60 "onnxruntime-gpu>=1.22,<1.23" "tensorrt-cu12-libs>=10.9,<10.10" "tensorrt-cu12-bindings>=10.9,<10.10" opencv-python-headless imageio-ffmpeg 2>&1 | tail -2; python -c "import cv2, onnxruntime" && break; echo "PIP_RETRY $t"; done
pip install -q --no-build-isolation -e . 2>&1 | tail -2; python -c "import streamsafe, cv2, onnxruntime" || { echo PIP_FAIL; exit 1; }
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }
ffmpeg -hide_banner -protocols 2>/dev/null | grep -E "^\s*(srt|rtmps)\s*$" | tr '\n' ' '; echo "<- protocols"
export BACKEND=onnx ONNX_DIR=/workspace/m/v1 TRT_CACHE=/workspace/trt HEAD_DIR=/workspace/h/runs/run48_lucia_runner/heads_lucia/ali
TRT=1 python -m streamsafe.prepare --models $ONNX_DIR 2>&1 | grep -v "timing cache" | tail -3
export TRT=1 TRT_BUILD=0 LOGO=$ENG/marketing/logo_rgba.png SCENE_IMAGE=$ENG/marketing/logo.png SLATE_IMAGE=$ENG/marketing/logo.png SCENE_TEXT= SLATE_TEXT=

echo "== A: raw push to the address WITHOUT /app (what Kick's dashboard shows), 12 s"
NOAPP=$(python3 -c "import os; u=os.environ['STREAM_URL']; print(u.replace(':443/app/','/').replace('/app/','/'))")
timeout 25 ffmpeg -hide_banner -loglevel error -nostats -progress pipe:1 -re -f lavfi -i "color=black:size=1280x720:rate=30" -f lavfi -i anullsrc=r=48000:cl=stereo \
  -t 12 -c:v libx264 -preset veryfast -b:v 2500k -g 60 -c:a aac -f flv "$NOAPP" > $R/a_noapp.txt 2>&1; echo "A_EXIT $?"
grep -E "^total_size" $R/a_noapp.txt | tail -1; grep -v -E "^(frame|fps|stream_|bitrate|total_size|out_time|dup_|drop_|speed|progress)=" $R/a_noapp.txt | tail -3
sleep 20  # let the platform end that session

echo "== B: the app's Test connection on the full address"
python -c "import os, sys; from streamsafe.app import linktest; sys.exit(linktest.main(['--output', os.environ['STREAM_URL'], '--seconds', '6']))" > $R/b_linktest.txt 2>&1
echo "B_EXIT $?"; grep -E "LINK_TEST|STATUS" $R/b_linktest.txt | tail -6
sleep 20

echo "== C: live run, ${MINUTES} min: OBS off at ${OBS_OFF_S}s for 15 s, platform link cut at ${NET_CUT_S}s"
python -u -c "import runpy, signal, sys; signal.signal(signal.SIGINT, signal.default_int_handler); sys.argv[0] = 'streamsafe'; runpy.run_module('streamsafe', run_name='__main__')" \
  --input "srt://127.0.0.1:9000?mode=listener" --output "$STREAM_URL" --mode realtime --size 1920x1080 --delay 2 \
  --report-dir $R --stats $R/c_stats.json > $R/c_engine.log 2>&1 &
EP=$!
for i in $(seq 1 120); do grep -q '"waiting_obs"' $R/c_engine.log && break; sleep 1; done
echo "listener up after ${i}s"
obs() { ffmpeg -hide_banner -loglevel error -re -stream_loop -1 -i "$T1" -c:v libx264 -preset veryfast -b:v 6000k -g 60 -c:a aac -b:a 160k \
  -f mpegts "srt://127.0.0.1:9000?mode=caller" > $R/c_obs_$1.txt 2>&1 & echo $!; }
OP=$(obs 1); T0=$(date +%s)
cut_done=0; off_done=0
while kill -0 $EP 2>/dev/null; do
  now=$(( $(date +%s) - T0 ))
  if [ $off_done = 0 ] && [ $now -ge $OBS_OFF_S ]; then echo "t=$now OBS stops"; kill $OP; sleep 15; OP=$(obs 2); echo "t=$((now+15)) OBS back"; off_done=1; fi
  if [ $cut_done = 0 ] && [ $now -ge $NET_CUT_S ]; then echo "t=$now platform link cut"; pkill -f -- "-f flv" ; cut_done=1; fi
  if [ $now -ge $((MINUTES * 60)) ]; then echo "t=$now stop (Ctrl+C like the Stop button)"; kill -INT $EP; break; fi
  sleep 2
done
wait $EP; echo "ENGINE_EXIT $?"; kill $OP 2>/dev/null
grep -E "^STATUS" $R/c_engine.log | cut -c1-200
grep -E "LIVE_RESULT" $R/c_engine.log | python3 -c "
import json,sys
for l in sys.stdin:
    s=json.loads(l.split(' ',1)[1])['summary']
    print({k:s.get(k) for k in ('frames_in','frames_out','dropped_frames','failclosed_frames','slate_episodes','decisions_per_s','gpu','analyzer_errors')})"
ls $R; cat $R/session_*.txt 2>/dev/null | head -30
echo "== DONE $(date -u)"
