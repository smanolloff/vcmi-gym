#!/bin/bash

set -euxo pipefail

VASTAI_INSTANCE_ID=$(cat ~/.vast_containerlabel | cut -c3-)

function http() {
  curl --fail-with-body -H "Authorization: Bearer $VASTAI_API_KEY" \
    --url "https://console.vast.ai/api/v0/instances/$VASTAI_INSTANCE_ID" \
    -X "$1" --json "$2"
}

checkresult=$(cat /checkresult || :)

if [ "$checkresult" = "1" ]; then
  http PUT '{"label": "passed"}'
else
  http PUT '{"label": "failed"}' || :

  if [ "${1:-}" = "-d" ]; then
    http DELETE '{}'  # destroy instance
  fi
fi
