#!/bin/bash
# Promo video: intro card -> the protected clip (StreamSafe output, small logo corner mark) -> outro card. 1920x1080 30 fps.
#   bash marketing/make_promo.sh protected.mp4 out.mp4
set -eu
D=$(cd "$(dirname "$0")" && pwd); IN=$1; OUT=$2
ffmpeg -nostdin -y -loglevel error \
  -loop 1 -t 3 -framerate 30 -i "$D/intro.png" \
  -i "$IN" \
  -loop 1 -t 4.5 -framerate 30 -i "$D/outro.png" \
  -loop 1 -framerate 30 -i "$D/wm.png" \
  -f lavfi -t 1 -i anullsrc=r=48000:cl=stereo \
  -filter_complex "\
[0:v]scale=1920:1080,format=yuv420p,fade=t=in:st=0:d=0.6,fade=t=out:st=2.5:d=0.5,setsar=1[a];\
[1:v]fps=30,scale=1920:1080:flags=lanczos,setsar=1[m0];\
[3:v]format=rgba,colorchannelmixer=aa=0.8[wm];\
[m0][wm]overlay=W-w-36:30:shortest=1,format=yuv420p,fade=t=in:st=0:d=0.4[m];\
[2:v]scale=1920:1080,format=yuv420p,fade=t=in:st=0:d=0.6,setsar=1[c];\
[a][m][c]concat=n=3:v=1:a=0[v]" \
  -map "[v]" -c:v libx264 -preset slow -crf 17 -profile:v high -pix_fmt yuv420p -movflags +faststart -r 30 "$OUT"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT"
