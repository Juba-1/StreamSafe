#!/bin/bash
# Run a GPU test job and ALWAYS stop the pod afterwards (success, failure or crash), with nobody watching.
#   bash pod_guard.sh /r.sh            (the pod's start command)
# A stopped pod bills only its disk; deleting it stays a manual step. Never leave a pod on `sleep infinity`.
MARK=/workspace/.job_done
if [ -f "$MARK" ]; then echo "GUARD: job already ran once on this pod, not running it again"; fi
stop_pod() {
  echo "GUARD: stopping pod ${RUNPOD_POD_ID:-?} ($(date -u))"
  if command -v runpodctl >/dev/null && runpodctl stop pod "$RUNPOD_POD_ID"; then return; fi
  if [ -n "${RUNPOD_API_KEY:-}" ] && curl -fsS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
       "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"; then return; fi
  echo "GUARD: could not stop through the API; idling 10 min, then exiting the container"
  sleep 600; kill -TERM 1
}
trap stop_pod EXIT
[ -f "$MARK" ] || { timeout "${GUARD_MAX_S:-5400}" bash "$1"; echo "GUARD: job exit $?"; mkdir -p /workspace; touch "$MARK"; }
