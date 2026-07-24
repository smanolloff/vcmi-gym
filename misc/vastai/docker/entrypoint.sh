#!/bin/bash

if ! [ -f /workspace/.init ]; then
    bash /workspace/vcmi-gym/misc/vastai/docker/init.sh
    touch /workspace/.init
fi

/opt/instance-tools/bin/entrypoint.sh "$@"
