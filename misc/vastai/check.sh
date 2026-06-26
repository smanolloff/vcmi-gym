#!/bin/bash

set -euxo pipefail

TAG=false    # -t
DEL=false    # -d
FORCE=false  # -f

INIT_SECONDS=90     # cold start (40 envs, load weights, etc.)
ROLLOUT_SECONDS=29  # "ok" duration of 1 rollout
N_ROLLOUTS=5

while getopts "tdfi:r:n:" opt; do
    case "$opt" in
    t) TAG=true ;;
    d) DEL=true ;;
    f) FORCE=true ;;
    i) INIT_SECONDS=$OPTARG ;;
    r) ROLLOUT_SECONDS=$OPTARG ;;
    n) N_ROLLOUTS=$OPTARG ;;
    *) echo "Usage: $0 [-td] [-i INT] [-r INT] [-n INT]"; exit 1 ;;
    esac
done

if [ -e /workspace/.check ] && !$FORCE; then
    echo "Already checked, nothing to do."
    exit 0
fi


function http() {
  curl --fail-with-body -H "Authorization: Bearer $VAST_API_KEY" \
    --url "https://console.vast.ai/api/v0/instances/$VASTAI_INSTANCE_ID" \
    -X "$1" --json "$2"
}

SECONDS_CHECK=$((N_ROLLOUTS * ROLLOUT_SECONDS))
SECONDS_TIMEOUT=$((INIT_SECONDS + SECONDS_CHECK * 2))

CHECKPOINT=ipnkyfqb-1776005610
RUN_ID=${CHECKPOINT%-*}

function check() {
  . ~/.simorc
  cd /workspace/vcmi-gym

  if ! [ -f data/mppo-dna-heads/$CHECKPOINT-model-dna.pt ]; then
    download_checkpoint $CHECKPOINT
  fi

  link_checkpoint -y $CHECKPOINT $RUN_ID data/mppo-dna-heads/

  # timeout returns:
  #   0   command finished
  #   124 command timed out
  #   other = command failed
  command="python -m rl.v15.ppo_gnn --dry-run --max-rollouts $N_ROLLOUTS --skip-eval -f data/v15/$CHECKPOINT-config.json"
  date
  # Temporary disable -e to capture non-0 exit statuses without aborting
  set +e
  output=$(timeout --foreground $SECONDS_TIMEOUT $command)
  status=$?
  set -e
  date
  if [ $status -eq 124 ]; then
    echo "CHECK FAILED (timeout)"
    return $status
  elif [ $status -ne 0 ]; then
    echo "CHECK FAILED (error)"
    return $status
  fi

  line=$(printf "%s" "$output" | grep '"event": "finish"') || :
  t=$(printf "%s" "$line" | jq -e ".message.timers.all | round")

  if [ "$t" -le $SECONDS_CHECK ]; then
    echo "CHECK PASSED (t=$t, CHECK=$SECONDS_CHECK)"
    return 0
  else
    echo "CHECK FAILED (t=$t, CHECK=$SECONDS_CHECK)"
    return 1
  fi
}

$TAG && http PUT '{"label": "check..."}' || :

if check; then
    $TAG && http PUT '{"label": "PASSED"}' || :
    echo 1 > /workspace/.check
else
    $TAG && http PUT '{"label": "FAILED"}' || :
    $DEL && http DELETE '{}' || :
    echo 0 > /workspace/.check
fi
