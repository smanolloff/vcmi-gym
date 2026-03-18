#!/bin/bash

set -euxo pipefail

VASTAI_API_KEY=
BENCHMARK_N_ROLLOUTS=5
BENCHMARK_SECONDS_MAX=$((BENCHMARK_N_ROLLOUTS * 28))

onstart="set -x"
onstart+="; cd /"
onstart+="; curl -LO https://raw.githubusercontent.com/smanolloff/vcmi-gym/refs/heads/main/misc/vastai/preinit.sh"
onstart+='; tmux new-session -d "bash -xc \"bash /preinit.sh; bash /init.sh; bash /check.sh; bash/resolve.sh -d; exec \\$SHELL\""'

RENT_BODY=$(jq -n --arg onstart "$onstart" '{
  "env": {
    "AWS_ACCESS_KEY": $ENV["VASTAI_AWS_ACCESS_KEY"],
    "AWS_SECRET_KEY": $ENV["VASTAI_AWS_SECRET_KEY"],
    "VCMI_ARCHIVE_KEY": $ENV["VASTAI_VCMI_ARCHIVE_KEY"],
    "WANDB_API_KEY": $ENV["VASTAI_WANDB_API_KEY"],
    "VASTAI_API_KEY": $ENV["VASTAI_BENCHMARK_API_KEY"]
  },
  "disk": 25.0,
  "onstart": $onstart,
  "cancel_unavail": true,
  "template_hash_id": "9535ff4084fd850b4c1cae890febf5e0"
}')

function http() {
  # VastAI API expects traling slash on some endpoints
  curl --fail-with-body -H "Authorization: Bearer $VASTAI_API_KEY" \
    -X "$1" --url "https://console.vast.ai/api/v0${2#/}/" --json "$3"
}

function create_instance() {
  http PUT /asks/$1 "$RENT_BODY"
}

