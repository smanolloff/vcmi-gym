#!/bin/bash

set -euxo pipefail

#
# Example:
# autorun.sh zvytfdpo pdpyqkrb eaqbvprl
#
USAGE='autorun.sh runid1 [runid2 ...]'

[ $# -gt 0 ] || { echo "No autoruns configured"; exit 0; }

source ~/.simorc

AUTORUNS=$(jq -Rnc --arg row "$*" '$row | split(" ")')

# Set label to wait
/opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID wait...

while true; do
    # List vastai instances
    instances=$(/opt/instance-tools/bin/vastai show instances --raw | jq '[.[] | {id: .id, status: (.actual_status // "-"), label: (.label // "-")} | select(.status | test("^(expired|exited|stopped)$") | not)]')

    # Check if this is the first waiting instance
    is_first=$(echo "$instances" | jq -r --arg id "$VASTAI_INSTANCE_ID" 'map(select(.label == "wait...")) | sort_by(.id) | first.id | tostring == $id')

    [ "$is_first" = "true" ] && break || sleep 60
done

# Find the first autorun which is not present in labels
labels=$(echo "$instances" | jq -c 'map(.label)')
autorun=$(jq -nr --argjson labels "$labels" --argjson autoruns "$AUTORUNS" 'first($autoruns[] | select(. as $x | $labels | index($x) | not)) // empty')

if [ -z "$autorun" ]; then
    # Set label to ready
    /opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID ready
else
    # XXX: no setting label to "$autorun" as the RL algo will set it
    # There is a race condition, but not a critical one
    tag=$(python -c '
from rl.v15.util.persistence import find_latest_tag
from rl.v15.util.structured_logger import StructuredLogger
import datetime as dt
tag, _ts = find_latest_tag(
    StructuredLogger(level=40, context=dict(name="test")),
    "ppo",
    "eaqbvprl",
    {"bucket_name": "vcmi-gym", "s3_dir": "v15/models"},
    dt.datetime(2000, 1, 1).astimezone(dt.timezone.utc))
print(tag)')

    download_checkpoint ppo $autorun-$tag
    link_checkpoint ppo $autorun-$tag $autorun
    train_gnn ppo $autorun
fi
