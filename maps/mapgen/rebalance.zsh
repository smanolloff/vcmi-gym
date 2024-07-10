#!/usr/bin/zsh

# Script for evaluating and optimizing a map
# Usage: rebalance.zsh <N_WORKERS> <WATCHDOGFILE> <MAP>
#        (from vcmi-gym root)

[ "$PWD" = "$VCMIGYM" ] || { echo "Script must be run from \$VCMIGYM dir"; exit 1; }

source .venv/bin/activate

# add timestamps to xtrace output
export PS4="+[%D{%Y-%m-%d %H:%M:%S}]<$$> (rebalance.zsh) "

set -eux
readonly N_WORKERS=$1
readonly WATCHDOGFILE=$2
readonly MAP=$3

[ -r "$MAP" ] || { echo "Bad map: $MAP"; exit 1; }
[[ "$N_WORKERS" =~ [1-9][0-9]* ]] || { echo "Bad N_WORKERS: $N_WORKERS"; exit 1; }
[ $N_WORKERS -gt 0 ] || { echo "Bad N_WORKERS: $N_WORKERS"; exit 1; }

readonly map=$(realpath "$MAP")
VCMIMAP=${MAP#$VCMIGYM/maps/}
readonly VCMIMAP=${VCMIMAP#maps/}

[ "$MAP" = "maps/$VCMIMAP" ] || { echo "script error"; exit 1; }

readonly DB="maps/mapgen/tmp/$(basename "$MAP" .vmap)-rebalance.sqlite3"

# DB must be initialized manually using the SQL files in vcmi/server/ML/sql
[ -r "$DB" ] || { echo "DB not found: $DB"; exit 1; }

# Function to run a command in the background and prefix output with job ID
bgjob() {
    local job_id=$1
    shift

    # Create a named pipe
    local fifo="${TMPDIR:-/tmp}/bgjob_${job_id}.fifo"
    rm -f $fifo
    mkfifo "$fifo"

    # Run the command in the background, redirecting its output to the named pipe
    {
            "$@"
            echo "done"
    } >"$fifo" 2>&1 &

    # Read from the named pipe and prefix each line with the job ID
    {
            set +x
            while IFS= read -r line; do
                    echo -en "\033[0m"
                    echo "<job=$job_id PID=$!> [$(date +"%Y-%m-%d %H:%M:%S")] $line"
            done <"$fifo"
            set -x
            # Remove the named pipe after the job is done
            rm -f "$fifo"
    } &
}

function run_mlclient() {
    for _ in $(seq 10); do
        $VCMI/rel/bin/mlclient-headless \
            --loglevel-ai error --loglevel-global error --loglevel-stats info \
            --random-heroes 1 --random-obstacles 1 --swap-sides 0 \
            --red-ai MMAI_SCRIPT_SUMMONER --blue-ai MMAI_SCRIPT_SUMMONER \
            --stats-mode red \
            --stats-storage "$DB" \
            --stats-timeout 300000 \
            --stats-persist-freq 0 \
            --max-battles 10000 \
            --map "$VCMIMAP"
    done
}

while true; do
    touch "$WATCHDOGFILE"

    for i in $(seq 0 $((N_WORKERS-1))); do
        # Use stats-sampling=max-battles+1 to enable stats sampling
        # by using only the distributions calculated after db was loaded
        # (each redistribution involves disk IO)
        bgjob $i run_mlclient
    done

    wait
    touch "$WATCHDOGFILE"
    python maps/mapgen/rebalance.py "$VCMIMAP" "$DB"
done
