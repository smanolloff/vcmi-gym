#!/bin/zsh

# XXX: This is a LINUX-specific version
#      for Mac, commands must be amended

#
# Watchdog script for maps/mapgen/rebalance.zsh
#
# Usage:
#
USAGE="
Usage: zsh maps/mapgen/watchdog.zsh <N_WORKERS> <MAP>
"

readonly N_WORKERS=$1
readonly MAP=$2

[ -n "$MAP" ] || { echo "$USAGE"; exit 1; }

# in minutes
readonly CHECK_EVERY=30

readonly IDENT="*** [🐕]"

function terminate_rebalance() {
  for i in $(seq 3); do
    pkill -g 0 -f rebalance.zsh || return 0  # no more procs => termination successful
    pkill -g 0 -f mlclient-headless || return 0  # no more procs => termination successful

    for j in $(seq 3); do
      sleep 3
      pkill -g 0 -f rebalance.zsh || return 0
      pgrep -g 0 -f mlclient-headless || return 0
    done
  done

  #
  # 3. Still alive => SIGKILL for ALL
  #
  for i in $(seq 3); do
    # Linux process kills itself with -g0...
    # It also names all sub-proccesses the same way
    pkill --signal=9 -g 0 -f rebalance.zsh || return 0  # no more procs => termination successful
    pkill --signal=9 -g 0 -f mlclient-headless || return 0

    for j in $(seq 3); do
      sleep 3
      pgrep -g 0 -f rebalance.zsh || return 0
      pgrep -g 0 -f mlclient-headless || return 0
    done
  done

  #
  # 4. STILL alive => abort
  #

  ps -ef | grep -E "mlclient-headless|rebalance.zsh"
  echo "$IDENT ERROR: failed to terminate processes"
  return 1
}

function handle_sigint() {
  echo "$IDENT SIGINT caught"
  terminate_rebalance
  exit 0
}

trap "handle_sigint" INT

# add timestamps to xtrace output
export PS4="+[%D{%Y-%m-%d %H:%M:%S}]<$$> (watchdog.zsh) "

set -eux

echo "$IDENT Watchdog PID: $$"

readonly WATCHDOGFILE=/tmp/watchdogfile_rebalance
touch $WATCHDOGFILE
ts=$(stat -c %Y $WATCHDOGFILE)
maps/mapgen/rebalance.zsh "$N_WORKERS" "$WATCHDOGFILE" "$MAP" &

while true; do
  ts0=$ts
  sleep $((CHECK_EVERY * 60))
  ts=$(stat -c %Y $WATCHDOGFILE)
  if [ $ts -eq $ts0 ]; then
    echo "$IDENT file not modified in the last ${CHECK_EVERY}m"
    terminate_rebalance
    maps/mapgen/rebalance.zsh "$N_WORKERS" "$WATCHDOGFILE" "$MAP" &
  fi
done
