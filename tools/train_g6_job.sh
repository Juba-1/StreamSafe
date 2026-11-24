#!/bin/bash
# Train heads with the GTA 6 round (labels bench/labels/$LABELS), compare with the product heads on the held-out
# GTA 6 crops and on held-out clips (covered person-decisions + old|new sheets). (Header shared with: Sensitivity check (GPU pod, through tools/pod_guard.sh): the real engine, offline, at the
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
        snapshot_download(R,repo_type='dataset',token=T,local_dir='/workspace/h',allow_patterns=['runs/run48_lucia_runner/heads_lucia/ali/*.txt','runs/run48_lucia_runner/lucia.npz','runs/run37_ali_labels_photos/labeled37.npz','runs/run37_ali_labels_photos/oi/photos_emb.npz','runs/run30_hard_mining/hard/meta.csv','runs/run30_hard_mining/hard/emb.npy','runs/mine_g6_r1/g6/meta.csv','runs/mine_g6_r1/g6/emb.npy','runs/mine_g6_r2/g6/meta.csv','runs/mine_g6_r2/g6/emb.npy','runs/train_g6_r1/heads_w1/ali/*.txt'])
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



pip install -q scikit-learn 2>&1 | tail -1
H=/workspace/h/runs; OLD=$HEAD_DIR; LABELS=${LABELS:-g6_r1_claude.csv}
echo "== g6 examples"
python - $H $ENG/bench/labels <<'P'
import csv, sys, numpy as np
H, LB = sys.argv[1], sys.argv[2]
# rounds: (mining run, labels); round 2 = Ali's labels, its test split is the GTA 6 test set
rounds = [("mine_g6_r1", "g6_r1_claude.csv"), ("mine_g6_r2", "g6_r2_ali.csv")]
parts = {"train": [], "test": []}
for run, lab in rounds:
    L = {int(r["id"]): r["label"].strip().upper() for r in csv.DictReader(open(f"{LB}/{lab}"))}
    meta = list(csv.DictReader(open(f"{H}/{run}/g6/meta.csv"))); E = np.load(f"{H}/{run}/g6/emb.npy").astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    for r in meta:
        c = L.get(int(r["id"]), "U")
        if c in ("R", "S", "M") and not (r["split"] == "test" and run != "mine_g6_r2"):
            parts[r["split"]].append((E[int(r["id"])], c, "gta6_" + r["video"].split("_")[0], float(r["h"])))
for split, rows in parts.items():
    X = np.stack([x for x, *_ in rows]).astype(np.float16); C = np.array([c for _, c, *_ in rows])
    np.savez(f"/workspace/g6_{split}.npz", X=X, C=C, G=np.array([g for *_, g, _ in rows]), H=np.array([h for *_, h in rows]))
    print(split, {c: int((C == c).sum()) for c in "RSM"})
P
for v in base:lucia w1:lucia,g6 w2:lucia,g6; do
  n=${v%%:*}; ex=$(echo ${v##*:} | sed "s#lucia#$H/run48_lucia_runner/lucia.npz#; s#g6#/workspace/g6_train.npz#")
  W=1; [ $n = w2 ] && W=2
  EXTRA=$ex EXTRA_WEIGHT=$W RECIPES=ali python -m training.compare_heads $R/heads_$n $H/run37_ali_labels_photos/labeled37.npz \
    $H/run37_ali_labels_photos/oi/photos_emb.npz $H/run30_hard_mining/hard $ENG/bench/labels/run30_hard_ali.csv $OLD 2>&1 | grep -E "HEADS_CMP ali|Error|error" | cut -c1-600
done
echo "== held-out GTA 6 crops"
python - $R $OLD <<'P'
import sys, json, numpy as np
from streamsafe.models.heads import RiskHeads
R, old = sys.argv[1], sys.argv[2]; d = np.load("/workspace/g6_test.npz"); X = d["X"].astype(np.float32); C = d["C"]
out = {}
for name, path in [("v1", old), ("v2_product", "/workspace/h/runs/train_g6_r1/heads_w1/ali")] + [(n, f"{R}/heads_{n}/ali") for n in ("base", "w1", "w2")]:
    h = RiskHeads.load(path); cov = h.prisk(X) >= h.thr
    out[name] = {"thr": round(float(h.thr), 4), **{c: f"{int(cov[C == c].sum())}/{int((C == c).sum())}" for c in "RSM"}}
txt = json.dumps(out, indent=1); print(txt); open(f"{R}/heldout.json", "w").write(txt)
P
echo "== held-out clips: product vs w2"
mkdir -p $R/sheets
for c in nightclub_09 beach_13 street_30 npc_21 drive_23; do
  ffmpeg -hide_banner -loglevel error -ss 30 -i /workspace/v/gta6/$c.mp4 -t 60 -an -c:v libx264 -preset veryfast -crf 20 -y /workspace/in_$c.mp4
  for n in product w1 w2; do
    hd=/workspace/h/runs/train_g6_r1/heads_w1/ali; [ $n != product ] && hd=$R/heads_$n/ali
    HEAD_DIR=$hd timeout 900 python -m streamsafe.cli --input /workspace/in_$c.mp4 --output /workspace/o_${c}_$n.mp4 --mode offline \
      --size 1280x720 --fps 15 --diag /workspace/d_${c}_$n.json --no-report > /workspace/e_${c}_$n.txt 2>&1 || echo "ENG_FAIL $c $n"
    python -c "import json;d=json.load(open('/workspace/d_${c}_$n.json'))['dec'];a=sum(len(x[1]) for x in d);b=sum(1 for x in d for y in x[1] if y[3]);print('$c $n persons',a,'covered',b)" | tee -a $R/clips.txt
  done
  rows=""
  for t in 5 15 25 35 45 55; do
    for n in product w1 w2; do ffmpeg -hide_banner -loglevel error -ss $t -i /workspace/o_${c}_$n.mp4 -frames:v 1 -vf scale=480:270 -y /workspace/f_${n}_$t.png; done
    ffmpeg -hide_banner -loglevel error -i /workspace/f_product_$t.png -i /workspace/f_w1_$t.png -i /workspace/f_w2_$t.png -filter_complex hstack=3 -y /workspace/row_$t.png
    rows="$rows -i /workspace/row_$t.png"
  done
  ffmpeg -hide_banner -loglevel error $rows -filter_complex vstack=6 -q:v 4 -y $R/sheets/$c.jpg && echo "SHEET $c"
done
echo "== DONE $(date -u)"
