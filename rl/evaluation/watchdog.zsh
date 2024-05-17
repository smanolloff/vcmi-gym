#!/bin/zsh

# XXX: This is a LINUX-specific version
#      for Mac, commands must be amended

#
# Usage:
#
USAGE="
Usage: zsh rl/evaluation/watchdog.zsh
"

IS_LINUX=false
if [ "$(uname)" = "Linux" ]; then
  IS_LINUX=true
fi

if [ -z "$CHECK_FIRST" ]; then
  CHECK_FIRST=1
fi

if [ -z "$CHECK_EVERY" ]; then
  CHECK_EVERY=10
fi

IDENT="*** [🐕]"

function terminate_python() {
  for i in $(seq 3); do
    pkill -g 0 -f python || return 0  # no more procs => termination successful

    for j in $(seq 3); do
      sleep 10
      pgrep -g 0 -f python || return 0
    done
  done

  #
  # 3. Still alive => SIGKILL for ALL
  #
  for i in $(seq 3); do
    # Linux process kills itself with -g0...
    # It also names all sub-proccesses the same way
    pkill --signal=9 -g 0 -f python || return 0

    for j in $(seq 3); do
      sleep 10
      pgrep -g 0 -f python || return 0
    done
  done

  #
  # 4. STILL alive => abort
  #

  ps -ef | grep "python"
  echo "$IDENT ERROR: failed to terminate processes"
  return 1
}

function handle_sigint() {
  echo "$IDENT SIGINT caught"
  terminate_python
  exit 0
}

trap "handle_sigint" INT

source .venv/bin/activate

# add timestamps to xtrace output
export PS4="+[%D{%Y-%m-%d %H:%M:%S}]<$$> "

set -eux

echo "$IDENT Watchdog PID: $$"

WATCHDOGFILE=/tmp/watchdogfile
touch WATCHDOGFILE

python rl/evaluation/evaluate.py $WATCHDOGFILE &
ts0=$(stat -c %Y $WATCHDOGFILE)

while true; do
  sleep $((CHECK_FIRST * 60))
  ts=$(stat -c %Y $WATCHDOGFILE)
  if [ $ts -eq $ts0 ];
    echo "$IDENT file not modified in the last ${CHECK_EVERY}m"
    terminate_python
    python rl/evaluation/evaluate.py $WATCHDOGFILE &
  else
    ts0=$ts
  fi
done
