#!/bin/bash

#
# Prepares a "blank" VastAI VM for init:
# - creates .env
# - creates init.sh (does NOT execute it)
#

set -euxo pipefail

cd /

cat <<-EOF >.env
AWS_ACCESS_KEY='$AWS_ACCESS_KEY'
AWS_SECRET_KEY='$AWS_SECRET_KEY'
VCMI_ARCHIVE_KEY='$VCMI_ARCHIVE_KEY'
WANDB_API_KEY='$WANDB_API_KEY'
EOF

for script in init check resolve; do
    curl -sLO https://raw.githubusercontent.com/smanolloff/vcmi-gym/refs/heads/main/misc/vastai/$script.sh
done
