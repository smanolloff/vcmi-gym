#!/bin/bash

set -euxo pipefail

N_ROLLOUTS=5
ROLLOUT_SECONDS=27
SECONDS_MAX=$((N_ROLLOUTS * ROLLOUT_SECONDS))

cd /workspace/vcmi-gym

download_checkpoint ytoowqgj-1773645017
link_checkpoint ytoowqgj-1773645017 ytoowqgj data/mppo-dna-heads/

# timeout returns:
#   0   command finished
#   124 command timed out
#   other = command failed
command="python -m rl.algos.mppo_dna_gnn.mppo_dna_gnn --dry-run --rollouts 5 --skip-eval"
output=$(timeout -f $((SECONDS_MAX * 2)) $command)
[ $? -eq 0 ] || { echo "CHECK FAILED (timeout)"; exit 1; }

if printf "%s" "$output" | grep '"event": "finish"' | jq -e ".message.timers.all < $SECONDS_MAX"; then
  echo "CHECK PASSED"
  echo 1 > /checkresult
else
  echo "CHECK FAILED"
  echo 0 > /checkresult
fi
