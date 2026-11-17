#!/bin/bash
# Launch-day check on one GPU pod (through tools/pod_guard.sh): the live end-to-end run on the real platform
# (STREAM_URL from a RunPod secret), then speed/load/stress (bench/laptop_job.sh with PLAN).
#   RUN_TAG=launch_r1 [MINUTES=8] [PLAN="short crowd fps60 long"] bash tools/launch_check_job.sh
set -u
TAG=${RUN_TAG:?}; ENG=$(cd "$(dirname "$0")/.." && pwd)
if [ -n "${STREAM_URL:-}" ]; then
  RUN_NAME=${TAG}_live MINUTES=${MINUTES:-8} timeout 2400 bash "$ENG/tools/live_e2e_job.sh"; echo "LIVE_EXIT $?"
  mv /workspace/res /workspace/res_live 2>/dev/null
else echo "LIVE_SKIPPED (no STREAM_URL)"; fi
env -u STREAM_URL RUN_NAME=${TAG}_perf PLAN="${PLAN:-short crowd fps60 long}" bash "$ENG/bench/laptop_job.sh"; echo "PERF_EXIT $?"
