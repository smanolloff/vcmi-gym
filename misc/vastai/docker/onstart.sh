#!/bin/bash
# Logs are in /var/log/onstart.log
service cron start || :
tmux new-session -d "set -x; bash /workspace/vcmi-gym/misc/vastai/docker/init.sh; exec $SHELL"
