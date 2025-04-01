#!/bin/bash

set -x

while true; do
  "$@"
  retval=$?
  if [ $retval -eq 0 ]; then
    exit 0
  fi

  cat <<-EOF
Bad exit code: $retval
Restarting in 5 seconds...
EOF

  # 5 seconds for ^C, otherwise restart
  sleep 5
done

