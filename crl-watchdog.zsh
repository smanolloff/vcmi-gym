#!/bin/zsh
# XXX: Not using fish due to lack of xtrace in trap functions

#
# Usage:
#
USAGE="
Usage:

zsh crl-watchdog.zsh -R ACTION GROUP_ID RUN_ID
<OR>
zsh crl-watchdog.zsh ACTION path/to/config.yml
<OR>
CHECK_FIRST=10 CHECK_EVERY=30 zsh crl-watchdog.zsh ACTION path/to/config.yml
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
  # 2. Alive after 10s => SIGTERM for vcmi_gym.tools.main_crl
  #

  # pkill -g 0

  for i in $(seq 3); do
    pkill -g 0 -f vcmi_gym.tools.main_crl || return 0  # no more procs => termination successful

    for j in $(seq 3); do
      sleep 10
      pgrep -g 0 || return 0
    done
  done

  #
  # 3. Still alive => SIGTERM for ALL
  #
  for i in $(seq 3); do

    if $IS_LINUX; then
      # Linux process kills itself with -g0...
      # It also names all sub-proccesses the same way
      # Apparently, 3 of them always get unresponsive and must be KILLed
      pkill --signal=9 -g 0 -f vcmi_gym.tools.main_crl || return 0
    else
      pkill -g 0 || return 0  # no more procs => termination successful
    fi

    for j in $(seq 3); do
      sleep 10
      pgrep -g 0 || return 0
    done
  done

  #
  # 4. STILL alive => abort
  #

  # XXX: for Linux, pgrep -g0 returns itself and there is no easy way to
  # reliably check if all subprocesses were killed.
  if $IS_LINUX; then
    # XXX: the -f is crucial here: it filters out our own process to prevent suicide
    pkill --signal=9 -g 0 -f python
    return 0
  fi

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
  find "$1" -name 'events.out.tfevents.*' -mmin -${2} | grep -q .
}

trap "handle_sigint" INT

source .venv/bin/activate

# add timestamps to xtrace output
export PS4="+[%D{%Y-%m-%d %H:%M:%S}]<$$> "

set -eux

#
#
#

args=()

if [ "$1" = "-R" ]; then
  # Resume
  action=$2
  group=$3
  run=$4
  out_dir_template="data/{group_id}/{run_id}"
  args+=("-R")
else
  # Init
  action=$1
  cfg=$2
  [ -r "$cfg" ]
  group=$(read_cfg ".group_id" "$cfg")
  run=$(read_cfg ".run_id" "$cfg")
  out_dir_template=$(read_cfg ".out_dir_template" "$cfg")
  args+=("-c" "$cfg")
fi

if [ -z "$action" \
    -o -z "$group" \
    -o -z "$run" \
    -o "$out_dir_template" != "data/{group_id}/{run_id}" ]; then
  echo "$USAGE"
  exit 1
fi

args+=("-g" "$group")
args+=("-r" "$run")
args+=("$action")

out_dir="${out_dir_template/\{group_id\}/$group}"
out_dir="${out_dir/\{run_id\}/$run}"

echo "$IDENT Watchdog PID: $$"

python -m vcmi_gym.tools.main_crl "${args[@]}" &
sleep $((CHECK_FIRST * 60))
if ! find_recent_tflogs "$out_dir" $CHECK_FIRST; then
  echo "$IDENT boot failed"
  terminate_vcmi_gym
  # Startup failure is either a syntax error or a wandb "resume" exception
  # We must not restart with "-R", as it may incorrectly resume a previous run
  exit 1
fi

while sleep $((CHECK_EVERY * 60)); do
  if ! find_recent_tflogs "$out_dir" $CHECK_EVERY; then
    echo "$IDENT No new tfevents in the last ${CHECK_EVERY}m"
    terminate_vcmi_gym
    python -m vcmi_gym.tools.main_crl -R -g "$group" -r "$run" $action &
  fi
done
