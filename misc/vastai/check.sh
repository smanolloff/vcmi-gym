#!/bin/bash

set -euxo pipefail

N_ROLLOUTS=5
INIT_SECONDS=30
ROLLOUT_SECONDS=27
SECONDS_OK=$((N_ROLLOUTS * ROLLOUT_SECONDS))
SECONDS_TIMEOUT=$((INIT_SECONDS + SECONDS_MAX * 2))

. ~/.simorc
cd /workspace/vcmi-gym

download_checkpoint ytoowqgj-1773645017
link_checkpoint ytoowqgj-1773645017 ytoowqgj data/mppo-dna-heads/

# timeout returns:
#   0   command finished
#   124 command timed out
#   other = command failed
command="python -m rl.algos.mppo_dna_gnn.mppo_dna_gnn --dry-run --max-rollouts $N_ROLLOUTS --skip-eval"
set +e
output=$(timeout --foreground $SECONDS_TIMEOUT $command)
if [ $? -nq 0 ]; then
  echo "CHECK FAILED (timeout)"
  echo 0 > /checkresult
  exit 1
fi

if printf "%s" "$output" | grep '"event": "finish"' | jq -e ".message.timers.all < $SECONDS_MAX"; then
  echo "CHECK PASSED"
  echo 1 > /checkresult
else
  echo "CHECK FAILED"
  echo 0 > /checkresult
fi
