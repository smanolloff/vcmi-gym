#!/bin/bash

set -euxo pipefail

TAG=false   # -t
DEL=false   # -d

while getopts "td" opt; do
    case "$opt" in
    t) TAG=true ;;
    d) DEL=true ;;
    *) echo "Usage: $0 [-td]"; exit 1 ;;
    esac
done

function http() {
  curl --fail-with-body -H "Authorization: Bearer $VASTAI_API_KEY" \
    --url "https://console.vast.ai/api/v0/instances/$VASTAI_INSTANCE_ID" \
    -X "$1" --json "$2"
}

N_ROLLOUTS=5
INIT_SECONDS=90     # cold start (40 envs, load weights, etc.)
ROLLOUT_SECONDS=27  # "ok" duration of 1 rollout

SECONDS_CHECK=$((N_ROLLOUTS * ROLLOUT_SECONDS))
SECONDS_TIMEOUT=$((INIT_SECONDS + SECONDS_CHECK * 2))

CHECKPOINT=ytoowqgj-1773645017
RUN_ID=${CHECKPOINT%-*}

$TAG && http PUT '{"label": "check..."}' || :

. ~/.simorc
cd /workspace/vcmi-gym

if ! [ -f data/mppo-dna-heads/$CHECKPOINT-model-dna.pt ]; then
  download_checkpoint ytoowqgj-1773645017
fi

link_checkpoint -y ytoowqgj-1773645017 ytoowqgj data/mppo-dna-heads/

# timeout returns:
#   0   command finished
#   124 command timed out
#   other = command failed
command="python -m rl.algos.mppo_dna_gnn.mppo_dna_gnn --dry-run --max-rollouts $N_ROLLOUTS --skip-eval -f data/mppo-dna-heads/$CHECKPOINT-config.json"
set +e
date
output=$(timeout --foreground $SECONDS_TIMEOUT $command)
status=$?
date
if [ $status -ne 0 ]; then
  echo "CHECK FAILED (timeout)"
  echo 0 > /checkresult
  exit $status
fi

line=$(printf "%s" "$output" | grep '"event": "finish"') || :

if printf "%s" "$line" | jq -e ".message.timers.all < $SECONDS_CHECK"; then
  echo "CHECK PASSED"
  $TAG && http PUT '{"label": "passed"}' || :
  echo 1 > /checkresult
else
  echo "CHECK FAILED"
  $TAG && http PUT '{"label": "failed"}' || :
  $DEL && http DELETE '{}' || :
  echo 0 > /checkresult
fi
