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
readonly VCMIMAP=${MAP#$VCMIGYM/maps/}

[ "$MAP" = "$VCMIGYM/maps/$VCMIMAP" ] || { echo "script error"; exit 1; }

readonly DB_BASE="$(dirname "$MAP")/$(basename "$MAP" .vmap)"
readonly DB_COMMON="$DB_BASE-all.sqlite3"

function buildsql() {
  # this is is not used with just 1 worker
  [ $N_WORKERS -gt 1 ] || return 0;

  echo "ATTACH DATABASE '$DB_COMMON' AS db;"
  for i in $(seq 0 $((N_WORKERS-1))); do
    echo "ATTACH DATABASE '$DB_BASE-$i.sqlite3' AS db$i;"
  done

  echo "WITH united AS ("
  # XXX: start from 1 as the 0th db will be copied to DB_COMMON
  for i in $(seq 1 $((N_WORKERS-2))); do
    echo "  SELECT id, wins, games FROM db$i.stats WHERE games > 0 UNION ALL"
  done

  cat <<-SQL
  SELECT id, wins, games FROM db$((N_WORKERS-1)).stats WHERE games > 0
),
grouped AS (
  select id, sum(wins) as wins, sum(games) as games
  from united group by id
)
update db.stats
set wins = db.stats.wins + grouped.wins,
    games = db.stats.games + grouped.games
from grouped
where db.stats.id = grouped.id;
SQL
}

# Function to run a command in the background and prefix output with job ID
bgjob() {
    local job_id=$1
    shift

    # Create a named pipe
    local fifo="${TMPDIR:/tmp}/bgjob_${job_id}.fifo"
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
            echo "<job=$job_id PID=$!> $line"
        done <"$fifo"
        set -x
        # Remove the named pipe after the job is done
        rm "$fifo"
    } &
}

# Export the function so it can be used in subshells
export -f bgjob

readonly MERGESQL="$DB_BASE-merge.sql"
rm -f "$MERGESQL"

buildsql > "$MERGESQL"

while true; do
  touch "$WATCHDOGFILE"
  for i in $(seq 0 $((N_WORKERS-1))); do
    db="$DB_BASE-$i.sqlite3"

    if [ -r "$DB_COMMON" ]; then
      cp "$DB_COMMON" "$db"
    fi

    # Use stats-sampling=max-battles+1 to enable stats sampling
    # by using only the distributions calculated after db was loaded
    # (each redistribution involves disk IO)
    bgjob $i $VCMI/build/bin/mlclient-headless \
      --loglevel-ai error --loglevel-global error --loglevel-stats info \
      --random-heroes 1 --random-obstacles 1 --swap-sides 0 \
      --red-ai MMAI_SCRIPT_SUMMONER --blue-ai MMAI_SCRIPT_SUMMONER \
      --stats-mode red \
      --stats-storage "$db" \
      --stats-persist-freq 0 \
      --stats-sampling 100001 \
      --max-battles 100000 \
      --map "$VCMIMAP"
  done

  wait
  touch "$WATCHDOGFILE"
  cp "$db" "$DB_COMMON"
  sqlite3 "$DB_COMMON" < "$MERGESQL";
  python maps/mapgen/rebalance.py "$MAP" "$DB_COMMON"
done
