#!/bin/bash

#
# Prepares a "blank" VastAI VM for init:
# - creates .env
# - downloads init.sh, check.sh (does NOT execute them)
#

set -euxo pipefail

cd /workspace

cat <<-EOF >.env
AWS_ACCESS_KEY='$AWS_ACCESS_KEY'
AWS_SECRET_KEY='$AWS_SECRET_KEY'
VCMI_ARCHIVE_KEY='$VCMI_ARCHIVE_KEY'
WANDB_API_KEY='$WANDB_API_KEY'
EOF

for script in init check; do
    curl -sLO https://raw.githubusercontent.com/smanolloff/vcmi-gym/refs/heads/main/misc/vastai/$script.sh
done
