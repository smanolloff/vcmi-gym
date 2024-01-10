#!/bin/zsh
# XXX: Not using fish due to lack of xtrace in trap functions

#
# Usage:
#
#     zsh sb3_watchdog.zsh [group_id] [run_id] [loadfile]
#
# Example:
#
#     zsh sb3_watchdog.zsh ahcgnv23
#

CHECK_EVERY=300  # seconds

function terminate_vcmi_gym() {
  # XXX: disabled (SIGINT is ignored by background jobs!)
  # To make them receive it, job control must be enabled in the script:
  # - this is impossible with fish
  # - with zsh is possible, but puts the sub-process in its own process group
  #   => can't properly terminate all child processes...
  #   => just use sigterm and rely on python's atexit handlers for cleanup
  # #
  # # 1. Ask politely (SIGINT)
  # #
  # local pid=$1
  #
  # # The thread pool executor absorbs the first SIGINT
  # # => try twice in case we are restarting during make_vec_env_parallel()
  # for i in $(seq 2); do
  #   kill -INT $pid
  #   for j in $(seq 10); do
  #     sleep 1
  #     kill -0 $pid || return 0
  #   done
  # done

  #
  # 2. Alive after 10s => SIGTERM for vcmi-gym.py
  #

  # pkill -g 0
  pkill -g 0 -f vcmi-gym.py

  for i in $(seq 60); do
    sleep 1
    pgrep -g 0 || return 0  # no more procs => termination successful
  done

  ps aux | grep vcmi-gym.py

  #
  # 3. STILL alive after 10s => abort
  #
  echo "*** [🐕] ERROR: failed to terminate processes after 60s"
  exit 1
}

function handle_sigint() {
  echo "*** [🐕] SIGINT caught"
  terminate_vcmi_gym
  exit 0
}


trap "handle_sigint" INT

source .venv/bin/activate

set -x
group=$1
run=$2
loadfile=$3

if [ -z "$group" ]; then
  group=$(head /dev/urandom | LC_ALL=C tr -dc 'a-z0-8' | head -c 8)
  run=$group
fi

if [ -z "$run" ]; then
  run=$(head /dev/urandom | LC_ALL=C tr -dc 'a-z0-8' | head -c 8)
fi

while true; do
  if [ -r "$loadfile" ]; then
    python vcmi-gym.py train_mppo $group $run "$loadfile" &
  else
    python vcmi-gym.py train_mppo $group $run &
  fi

  while sleep $CHECK_EVERY; do
    # no tfevents => no training
    if ! find data/MPPO-$group/$run -name 'events.out.tfevents.*' -mtime -${CHECK_EVERY}s | grep -q . ; then
      echo "*** [🐕] No new tfevents ${CHECK_EVERY}s"
      terminate_vcmi_gym
      loadfile=data/MPPO-$group/$run/model.zip
      run=${a:0:8}_$(date +%s)
      # re-use the same run ID (ie. just continue training from last save)
      # set -g run (head /dev/urandom | LC_ALL=C tr -dc 'a-z0-8' | head -c 8)
      break
    fi
  done
done
