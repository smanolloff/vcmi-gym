#!/bin/bash

exec >>/root/onstart.log 2>&1
set -euxo pipefail

source ~/.simorc

if ! [ -f /workspace/.init ]; then
    /opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID init...
    bash /workspace/vcmi-gym/misc/vastai/docker/init.sh
    touch /workspace/.init
    /opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID ready
fi
