#!/bin/zsh
# XXX: Not using fish due to lack of xtrace in trap functions

#
# Usage:
#
#     zsh sb3_watchdog.zsh [group_id [run_id [loadfile]]]
#   <OR>
#     zsh sb3_watchdog.zsh [-c CONFIG]
#
# Examples:
#
#     zsh sb3_watchdog.zsh ahcgnv23
#     zsh sb3_watchdog.zsh ahcgnv23 fisughdk path/to/model.zip
#     zsh sb3_watchdog.zsh -c path/to/config.yml
#
# To generate a new group for all configs (config/train_mppo/{1,2,3}.yml):
#
# $ set -l group (head /dev/urandom | LC_ALL=C tr -dc 'a-z0-8' | head -c 8); \
#       and sed -i '' -r "s/^group_id:.+/group_id: $group/" config/train_mppo/{1,2,3}.yml
#

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

function gen_id() {
  head /dev/urandom | LC_ALL=C tr -dc 'a-z0-8' | head -c 8
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

trap "handle_sigint" INT

source .venv/bin/activate

set -eux

cfg=
group=
run=
loadfile=
iteration=

#
#
#
if [ "${1:-}" = "-c" ]; then
  cfg=$2
  test -z "${3:-}"

  group=$(read_cfg ".group_id" "$cfg")
  run=$(read_cfg ".run_id" "$cfg")
  loadfile=$(read_cfg ".model_load_file" "$cfg")
else
  group=${1:-}
  run=${2:-}
  loadfile=${3:-}
fi

if [ -z "$group" ]; then
  group=$(head /dev/urandom | LC_ALL=C tr -dc 'a-z0-8' | head -c 8)
fi

if [ -z "$run" ]; then
  run=$(head /dev/urandom | LC_ALL=C tr -dc 'a-z0-8' | head -c 8)
fi

# run_id must not contain these chars as per https://docs.wandb.ai/ref/python/init
[[ "$run" =~ '^[^/\#?%: ]+$' ]]

orig_run=$run

if [ "$loadfile" = "__latest__" ]; then
  loadfile=$(find_latest_loadfile)
fi

# run_id must be globally unique
while true; do
  run=${orig_run}-$(date +%s)
  # Multiple runs started in the same second might end up with the same name
  [ -d "data/$group/$run" ] || break
  sleep $((RANDOM%5 + 1))
done

mkdir -p data/$group/$run


while true; do
  echo -e "$IDENT group: $group\n$IDENT run: $run\n$IDENT loadfile: $loadfile"

  args=()

  [ -z "$cfg" ] || args+=("-c" "$cfg")
  [ -z "$iteration" ] || args+=(-i "$iteration")
  [ -z "$loadfile" ] || args+=("-l" "$loadfile")
  args+=(-g "$group")
  args+=(-r "$run")
  args+=(train_mppo)

  python vcmi-gym.py "${args[@]}" &

  while sleep $CHECK_EVERY; do
    # no tfevents => no training
    if ! find data/$group/$run -name 'events.out.tfevents.*' -mtime -${CHECK_EVERY}s | grep -q . ; then
      echo "$IDENT No new tfevents in the last ${CHECK_EVERY}s"
      terminate_vcmi_gym

      # Overwrite loadfile with the latest model snapshot, if any
      latest_loadfile=$(find_latest_loadfile)
      if [ -n "$latest_loadfile" ]; then
        loadfile=$latest_loadfile
        iteration=$(<${loadfile%/*}/iteration)
        let iteration++
      fi

      run=${orig_run}-$(date +%s)
      # re-use the same run ID (ie. just continue training from last save)
      # set -g run (head /dev/urandom | LC_ALL=C tr -dc 'a-z0-8' | head -c 8)
      break
    fi
  done
done
