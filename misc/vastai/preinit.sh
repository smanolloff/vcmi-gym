#!/bin/bash

#
# Prepares a "blank" VastAI VM for init:
# - calculates time drift and optionally sets tmux faketime alias
# - creates .env
# - downloads init.sh, check.sh (does NOT execute them)
#

set -euxo pipefail

# HTTP headers end with \r => grep printable chars
ref_date=$(curl -fsSI https://www.google.com | grep -Eio '^date:[[:print:]]+' | cut -d' ' -f2-)
ref_epoch=$(date -u -d "$ref_date" "+%s")
now_epoch=$(date +%s)

# There will always be some small diff (network delay) => trunc to minutes
diffmins=$(((ref_epoch - now_epoch) / 60))

# Add explicit "+"
[ $diffmins -gt 0 ] && offset="+${diffmins}m" || offset="${diffmins}m"

cd /workspace

export VASTAI_INSTANCE_ID=$(cat ~/.vast_containerlabel | cut -c3-)

cat <<-EOF >.env
AWS_ACCESS_KEY='$AWS_ACCESS_KEY'
AWS_SECRET_KEY='$AWS_SECRET_KEY'
VCMI_ARCHIVE_KEY='$VCMI_ARCHIVE_KEY'
WANDB_API_KEY='$WANDB_API_KEY'
VAST_API_KEY='$VAST_API_KEY'
VASTAI_INSTANCE_ID='$VASTAI_INSTANCE_ID'
EOF

if [ "$offset" !=  "0m" ]; then
    apt-get -o Acquire::Check-Date=false update
    apt-get -o Acquire::Check-Date=false -y install faketime
    faketime_so=$(dpkg -L libfaketime | grep libfaketime.so)

    export FAKETIME=$offset
    export LD_PRELOAD=$faketime_so
    export FAKETIME_DISABLE_SHM=1

    echo "FAKETIME=$offset" >> /etc/environment
    echo "LD_PRELOAD=$faketime_so" >> /etc/environment
    echo "FAKETIME_DISABLE_SHM=1" >> /etc/environment

    # Permanently enable faketime (effective globally and immediately)
    # XXX: breaks vastai's key exchange?
    # echo "$faketime_so" > /etc/ld.so.preload
fi

for script in init check; do
    curl -sLO https://raw.githubusercontent.com/smanolloff/vcmi-gym/refs/heads/main/misc/vastai/$script.sh
done

set -a
source .env
set +a
