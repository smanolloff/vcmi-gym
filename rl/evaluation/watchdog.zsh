#!/bin/zsh

# XXX: This is a LINUX-specific version
#      for Mac, commands must be amended

USAGE="
Usage: zsh rl/evaluation/watchdog.zsh N_WORKERS WORKER_ID
"

CHECK_EVERY=90  # minutes

N_WORKERS=$1
WORKER_ID=$2
WATCHDOGFILE=/tmp/evaluator-${WORKER_ID}

[ -n "$N_WORKERS" -a -n "$WORKER_ID" ] || { echo "$USAGE"; exit 1; }
[ $WORKER_ID -ge 0 ] || { echo "check failed: WORKER_ID >= 0"; exit 1; }
[ $WORKER_ID -lt $N_WORKERS ] || { echo "check failed: WORKER_ID < N_WORKERS"; exit 1; }

IDENT="*** [🐕]"

function terminate_evaluator() {
  for i in $(seq 3); do
    pkill -g 0 -f python || return 0  # no more procs => termination successful
    pkill -g 0 -f mlclient-headless || return 0  # no more procs => termination successful

    for j in $(seq 3); do
      sleep 3
      pkill -g 0 -f python || return 0
      pgrep -g 0 -f mlclient-headless || return 0
    done
  done

  #
  # 3. Still alive => SIGKILL for ALL
  #
  for i in $(seq 3); do
    # Linux process kills itself with -g0...
    # It also names all sub-proccesses the same way
    pkill --signal=9 -g 0 -f python || return 0  # no more procs => termination successful
    pkill --signal=9 -g 0 -f mlclient-headless || return 0

    for j in $(seq 3); do
      sleep 3
      pgrep -g 0 -f python || return 0
      pgrep -g 0 -f mlclient-headless || return 0
    done
  done

  #
  # 4. STILL alive => abort
  #

  ps -ef | grep -E "mlclient-headless|python"
  echo "$IDENT ERROR: failed to terminate processes"
  return 1
}

function handle_sigint() {
  echo "$IDENT SIGINT caught"
  terminate_evaluator
  exit 0
}

function start_evaluator() {
  python -m rl.evaluation.evaluate -w $WATCHDOGFILE -d locks.sqlite3 -I $N_WORKERS -i $WORKER_ID
}

trap "handle_sigint" INT

source .venv/bin/activate

# add timestamps to xtrace output
export PS4="+[%D{%Y-%m-%d %H:%M:%S}]<$$> "

set -eux

echo "$IDENT Watchdog PID: $$"
touch $WATCHDOGFILE
ts=$(stat -c %Y $WATCHDOGFILE)
sqlite3 locks.sqlite3 "CREATE TABLE IF NOT EXISTS LOCKS (id PRIMARY KEY)"
start_evaluator &

while true; do
  ts0=$ts
  sleep $((CHECK_EVERY * 60))
  ts=$(stat -c %Y $WATCHDOGFILE)
  if [ $ts -eq $ts0 ]; then
    echo "$IDENT $WATCHDOGFILE not modified in the last ${CHECK_EVERY}m"
    terminate_evaluator
    start_evaluator &
  fi
done

