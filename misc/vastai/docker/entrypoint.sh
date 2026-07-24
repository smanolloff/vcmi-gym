#!/bin/bash
service cron start
/opt/instance-tools/bin/entrypoint.sh "$@"
