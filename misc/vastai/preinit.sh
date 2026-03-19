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

cat <<-EOF >.env
AWS_ACCESS_KEY='$AWS_ACCESS_KEY'
AWS_SECRET_KEY='$AWS_SECRET_KEY'
VCMI_ARCHIVE_KEY='$VCMI_ARCHIVE_KEY'
WANDB_API_KEY='$WANDB_API_KEY'
EOF

if [ "$offset" !=  "0m" ]; then
    apt-get -o Acquire::Check-Date=false update
    apt-get -o Acquire::Check-Date=false -y install faketime
    faketime_so=$(dpkg -L libfaketime | grep libfaketime.so)
    # Permanently enable faketime (effective globally and immediately)
    echo "FAKETIME='${offset}'" >> /etc/environment
    echo "$faketime_so" > /etc/ld.so.preload
EOF
fi

for script in init check; do
    curl -sLO https://raw.githubusercontent.com/smanolloff/vcmi-gym/refs/heads/main/misc/vastai/$script.sh
done
