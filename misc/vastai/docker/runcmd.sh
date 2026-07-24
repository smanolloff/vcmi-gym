#!/bin/bash
trap 'echo "INTERRUPT DETECTED, EXIT 130"; exit 130' INT

df  # for logging
mb=$(df --output=avail -m / | tail -1 | awk '{print $1}')
if ! [[ $mb =~ ^[0-9]+$ ]]; then
    echo "Failed to determine free space: '$mb'"
    exit 1
fi

if [ $mb -lt 500 ]; then
    echo "Less than 500MB of free space: '$mb'"
    exit 1
fi

set +x
. /venv/main/bin/activate
set -x

# XXX: do NOT use exec here (does not call trap)
$@

echo "PROGRAM EXIT CODE: $?"
