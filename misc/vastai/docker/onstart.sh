#!/bin/bash
tmux new-session -d "bash -xc 'bash /workspace/vcmi-gym/misc/vastai/docker/init.sh; exec \$SHELL'"
