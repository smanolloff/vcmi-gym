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
    #
    # Measured values with 20K battles, 36 workers, 5.4K RPM HDD:
    # * 1 worker gathers data for 14min
    # * 36 workers need a total of 15min for dbupdate (~25s per worker)
    #
    # This is problematic as the 1st worker finishes a 2nd round
    # of data gathering, while the 36th worker still waits for dbupdate
    # from the 1st round.
    #
    # Increasing the number of battles causes a greater increase in data
    # gathering time compared to the increase in dbupdate time
    # => increase battles such that dbupdate time becomes the smaller value
    #
    # Expected values with 26K battles:
    # * 36 workers should need a total of 17min (+3s per per dbupdate)
    # * 1 worker should gather data for 19min
    #
    timeout_minutes=20  # XXX: ensure watchdog has bigger timeout

    for _ in $(seq 10); do
        touch "$WATCHDOGFILE"
        $VCMI/rel/bin/mlclient-headless \
            --loglevel-ai error --loglevel-global error --loglevel-stats info \
            --random-heroes 1 --random-obstacles 1 --swap-sides 0 \
            --red-ai MMAI_SCRIPT_SUMMONER --blue-ai MMAI_SCRIPT_SUMMONER \
            --stats-mode red \
            --stats-storage "$DB" \
            --stats-timeout $((timeout_minutes*60*1000)) \
            --stats-persist-freq 0 \
            --max-battles 26000 \
            --map "$VCMIMAP"
    done
}

for i in $(seq 1 10); do
    for j in $(seq 0 $((N_WORKERS-1))); do
        bgjob $j run_mlclient
    done

    wait
    cp "$DB" "${DB%.sqlite3}-$i.sqlite3"
    python maps/mapgen/rebalance.py "$VCMIMAP" "$DB"
    echo "update stats set wins=0, games=0" | sqlite3 "$DB"
done
