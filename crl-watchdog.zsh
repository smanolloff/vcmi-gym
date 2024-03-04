#!/bin/zsh
# XXX: Not using fish due to lack of xtrace in trap functions

#
# Usage:
#
USAGE="zsh crl_watchdog.zsh path/to/config.yml"

CHECK_FIRST=30
CHECK_EVERY=600  # seconds

IDENT="*** [🐕]"

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

  for i in $(seq 3); do
    pkill -g 0 -f vcmi-gym.py || return 0  # no more procs => termination successful

    for j in $(seq 3); do
      sleep 10
      pgrep -g 0 || return 0
    done
  done

  #
  # 3. Still alive => SIGTERM for ALL
  #
  for i in $(seq 3); do
    pkill -g 0 || return 0  # no more procs => termination successful

    for j in $(seq 3); do
      sleep 10
      pgrep -g 0 || return 0
    done
  done

  #
  # 4. STILL alive => abort
  #

  ps aux | grep "python@3.10"
  echo "$IDENT ERROR: failed to terminate processes"
  return 1
}

function handle_sigint() {
  echo "$IDENT SIGINT caught"
  terminate_vcmi_gym
  exit 0
}

function read_cfg() {
  local res=$(yq "$1" "$2")

  if [ "$res" = "null" -o "$res" = "~" ]; then
    echo ""
  else
    echo "$res"
  fi
}

function find_latest_loadfile() {
  [ -d "data/$group" ] || return 0

  find data/$group/$orig_run-* -type f -name 'model.zip' -exec stat -f "%Sm %N" -t "%s" {} \; \
    | sort -r \
    | head -1 \
    | awk '{print $2}'
}

function find_recent_tflogs() {
  find "$1" -name 'events.out.tfevents.*' -mtime -${2}s | grep -q .
}

trap "handle_sigint" INT

source .venv/bin/activate

# add timestamps to xtrace output
export PS4="+[%D{%Y-%m-%d %H:%M:%S}] "

set -eux

#
#
#

if [ "$#" -ne 1 ]; then
  echo "Usage: $USAGE"
  exit 1
fi

cfg=$1
group=$(read_cfg ".group_id" "$cfg")
run=$(read_cfg ".run_id" "$cfg")

out_dir_template=$(read_cfg ".out_dir_template" "$cfg")
out_dir="${out_dir_template/\{group_id\}/$group}"
out_dir="${out_dir/\{run_id\}/$run}"

args=()
args+=("-r" "$run")
args+=("-g" "$group")
args+=("crl_train_mppo")

echo "$IDENT Watchdog PID: $$"

python vcmi-gym.py -c "$cfg" "${args[@]}" &
sleep $CHECK_FIRST
if ! find_recent_tflogs "$out_dir" $CHECK_EVERY; then
  echo "$IDENT boot failed"
  terminate_vcmi_gym
  # Startup failure is either a syntax error or a wandb "resume" exception
  # We must not restart with "-R", as it may incorrectly resume a previous run
  exit 1
fi

while sleep $CHECK_EVERY; do
  if ! find_recent_tflogs "$out_dir" $CHECK_EVERY; then
    echo "$IDENT No new tfevents in the last ${CHECK_EVERY}s"
    terminate_vcmi_gym
    python vcmi-gym.py -R "${args[@]}" &
  fi
done
