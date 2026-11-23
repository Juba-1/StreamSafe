#!/bin/bash
# Why people are covered on the new GTA 6 clips: per person decision the risk score p; sheets of covered
# people (crop + p) and the p histogram, to set the slider levels and pick labels. (Was: Sensitivity check on new GTA 6 clips (GPU pod, through tools/pod_guard.sh): the real engine, offline, at the
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



echo "== why covered"
mkdir -p $R/sheets
for f in /workspace/v/gta6/*.mp4; do
  c=$(basename $f .mp4)
  THR_SCALE=1.0 timeout 900 python -m streamsafe.cli --input $f --secs ${SECS} --mode offline --size 960x540 --fps 10 \
    --diag /workspace/d_$c.json --stats /workspace/s_$c.json --no-report > /workspace/e_$c.txt 2>&1 || { echo "ENG_FAIL $c"; tail -3 /workspace/e_$c.txt; }
done
python - "$R" <<'P'
import json, glob, os, sys, cv2, numpy as np
R = sys.argv[1]; FPS = 10.0; AW, AH = 960, 540
rows, allp = [], []
for d in sorted(glob.glob('/workspace/d_*.json')):
    c = os.path.basename(d)[2:-5]
    dec = json.load(open(d))['dec']
    cap = cv2.VideoCapture(f'/workspace/v/gta6/{c}.mp4')
    picks, last = [], {}
    for fid, dets, *extra in dec:
        thr = extra[1] if len(extra) > 1 else None
        for tid, box, p, risky, _s in dets:
            allp.append((c, round(p, 3), int(risky), box[3] - box[1]))
            if risky and fid - last.get(tid, -99) >= 20:  # one crop per person every 2 s
                last[tid] = fid; picks.append((fid, box, p, thr))
    tiles = []
    for fid, box, p, thr in picks[:60]:
        cap.set(cv2.CAP_PROP_POS_MSEC, fid / FPS * 1000.0)
        ok, fr = cap.read()
        if not ok: continue
        fr = cv2.resize(fr, (AW, AH))
        x0, y0, x1, y1 = [int(v) for v in box]
        cr = fr[max(0, y0):max(y0 + 2, y1), max(0, x0):max(x0 + 2, x1)]
        cr = cv2.resize(cr, (120, 200))
        cv2.putText(cr, f"{p:.2f}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(cr)
    if tiles:
        while len(tiles) % 10: tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)])
        cv2.imwrite(f'{R}/sheets/{c}.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    rows.append(f"{c:14s} persons={sum(len(x[1]) for x in dec)} covered={sum(r for (cc, _, r, _) in allp if cc == c)} crops={len(tiles)}")
with open(f'{R}/p_all.csv', 'w') as f:
    f.write("clip,p,risky,h\n"); f.writelines(f"{a},{b},{r},{h}\n" for a, b, r, h in allp)
ps = np.array([p for _, p, r, _ in allp if r]) if allp else np.zeros(0)
hist = np.histogram(ps, bins=[0, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])[0].tolist() if len(ps) else []
txt = "\n".join(rows) + f"\ncovered p histogram (0 .2 .3 .4 .5 .6 .7 .8 .9 1): {hist}\n"
open(f'{R}/summary.txt', 'w').write(txt); print(txt)
P
echo "== DONE $(date -u)"
