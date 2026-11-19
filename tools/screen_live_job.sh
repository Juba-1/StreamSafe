#!/bin/bash
# Screen-share live test on a REAL platform (GPU pod, through tools/pod_guard.sh): a desktop (Xvfb) captured like
# OBS display capture -> the engine with the real models -> Kick + a viewers copy. While live: terminals, a web
# browser, the GTA 6 trailer in a player window and full screen, switching back and forth (xdotool).
#   RUN_NAME=screen_live_r1 [MINUTES=6] bash tools/screen_live_job.sh
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

echo "== desktop"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq xvfb openbox xterm mpv xdotool x11-utils fonts-noto-core >/dev/null 2>&1; echo "APT $?"
pip install -q playwright && python -m playwright install --with-deps chromium >/dev/null 2>&1; echo "PW $?"
Xvfb :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & sleep 2; export DISPLAY=:99
openbox >/dev/null 2>&1 & sleep 1
xterm -geometry 110x30+40+40 -fa Mono -fs 13 -bg '#111' -fg '#ddd' -e "bash -c 'top -d 1'" & sleep 1
CHROME=$(find /root/.cache/ms-playwright -name chrome -type f | head -1)

echo "== live screen share, ${MINUTES} min"
ffmpeg -hide_banner -loglevel warning -listen 1 -i rtmp://127.0.0.1:1935/live/demo -map 0 -c copy \
  -f tee "[f=flv:onfail=ignore]$STREAM_URL|[f=matroska]$R/viewer_view.mkv" > $R/relay.txt 2>&1 &
RP=$!; sleep 2
python -u -c "import runpy, signal, sys; signal.signal(signal.SIGINT, signal.default_int_handler); sys.argv[0] = 'streamsafe'; runpy.run_module('streamsafe', run_name='__main__')" \
  --input "srt://127.0.0.1:9000?mode=listener" --output "rtmp://127.0.0.1:1935/live/demo" --mode realtime --size 1920x1080 --delay 2 \
  --report-dir $R --stats $R/stats.json > $R/engine.log 2>&1 &
EP=$!
for i in $(seq 1 120); do grep -q '"waiting_obs"' $R/engine.log && break; sleep 1; done
# the "OBS display capture": the whole desktop, 30 fps, with silent audio
ffmpeg -hide_banner -loglevel error -f x11grab -framerate 30 -video_size 1920x1080 -i :99 -f lavfi -i anullsrc=r=48000:cl=stereo \
  -c:v libx264 -preset veryfast -tune zerolatency -b:v 8000k -g 60 -c:a aac -b:a 128k -f mpegts "srt://127.0.0.1:9000?mode=caller" > $R/capture.txt 2>&1 &
CP=$!; T0=$(date +%s)
el() { echo "t=$(( $(date +%s) - T0 )) $*"; }
sleep 20; el "terminal: typing"; xterm -geometry 100x20+700+400 -fa Mono -fs 14 -bg '#1e1e2e' -fg '#cdd6f4' -T chat & sleep 2
xdotool search --name chat windowactivate --sync 2>/dev/null; xdotool type --delay 90 "echo streaming GTA 6 with StreamSafe"; xdotool key Return; sleep 15
el "browser: the site"; "$CHROME" --no-sandbox --no-first-run --window-size=1500,950 --window-position=200,60 --user-data-dir=/tmp/ch https://streamsafe-sa.vercel.app >/dev/null 2>&1 & sleep 12
for s in 1 2 3 4 5 6; do xdotool key Page_Down; sleep 2; done
el "game: trailer in a window"; mpv --no-terminal --geometry=1280x720+300+150 --loop=inf --title=game "$T1" & MP=$!; sleep 40
el "game: full screen"; xdotool search --name game windowactivate --sync key f 2>/dev/null; sleep 60
el "switch to the browser"; xdotool search --name game key f 2>/dev/null; sleep 1; xdotool search --class chrome windowactivate 2>/dev/null; sleep 15
el "switch to the terminal"; xdotool search --name chat windowactivate 2>/dev/null; xdotool type --delay 90 "back to the game"; sleep 8
el "back to the game, full screen"; xdotool search --name game windowactivate --sync key f 2>/dev/null
while kill -0 $EP 2>/dev/null; do
  if [ $(( $(date +%s) - T0 )) -ge $((MINUTES * 60)) ]; then el "stop (like the Stop button)"; kill -INT $EP; break; fi
  sleep 2
done
wait $EP; echo "ENGINE_EXIT $?"; kill $CP $MP 2>/dev/null; sleep 3; kill -INT $RP 2>/dev/null; sleep 3
grep -E "^STATUS" $R/engine.log | cut -c1-160 | tail -8
cat $R/session_*.txt 2>/dev/null | head -14
ls -la $R; echo "== DONE $(date -u)"
