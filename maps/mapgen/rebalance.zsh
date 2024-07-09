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

mkdir -p maps/mapgen/tmp
readonly DB_BASE="maps/mapgen/tmp/$(basename "$MAP" .vmap)-rebalance"
readonly DB_COMMON="$DB_BASE-all.sqlite3"
readonly DB_LOCKS="$DB_BASE-locks.sqlite3"

rm -f "$DB_LOCKS"
touch "$DB_LOCKS"

function buildsql() {
  # this is is not used with just 1 worker
  [ $N_WORKERS -gt 1 ] || return 0;

  echo "ATTACH DATABASE '$DB_COMMON' AS db;"

  # XXX: exclude the last DB: it was copied to DB_COMMON
  maxmax=$((N_WORKERS-1))

  i=0
  while [ $i -lt $maxmax ]; do
    imax=$((i+9))
    [ $imax -le $maxmax ] || imax=$maxmax

    j=$i
    while [ $j -lt $imax ]; do
      echo "ATTACH DATABASE '$DB_BASE-$j.sqlite3' AS db$j;"
      let ++j
    done

    echo "WITH united AS ("
    j=$i
    while [ $j -lt $((imax-1)) ]; do
      echo "  SELECT id, wins, games FROM db$j.stats WHERE games > 0 UNION ALL"
      let ++j
    done

    cat <<-SQL
  SELECT id, wins, games FROM db$((imax-1)).stats WHERE games > 0
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

    j=$i
    while [ $j -lt $imax ]; do
      echo "DETACH DATABASE db$j;"
      let ++j
    done

    i=$imax
  done

}

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

readonly MERGESQL="$DB_BASE-merge.sql"
rm -f "$MERGESQL"

buildsql > "$MERGESQL"

while true; do
  touch "$WATCHDOGFILE"
  for i in $(seq 0 $((N_WORKERS-1))); do
    db="$DB_BASE-$i.sqlite3"

    if [ -r "$DB_COMMON" ]; then
      # XXX: `cp` is 30x slower when dst file exists for large files (1G)
      rm -f "$db"
      cp "$DB_COMMON" "$db"
    fi

    # Use stats-sampling=max-battles+1 to enable stats sampling
    # by using only the distributions calculated after db was loaded
    # (each redistribution involves disk IO)
    bgjob $i $VCMI/rel/bin/mlclient-headless \
      --loglevel-ai error --loglevel-global error --loglevel-stats info \
      --random-heroes 1 --random-obstacles 1 --swap-sides 0 \
      --red-ai MMAI_SCRIPT_SUMMONER --blue-ai MMAI_SCRIPT_SUMMONER \
      --stats-mode red \
      --stats-storage "$db" \
      --stats-persist-freq 0 \
      --stats-lockdb "$DB_LOCKS" \
      --stats-sampling 10001 \
      --max-battles 10000 \
      --map "$VCMIMAP"
  done

  wait
  touch "$WATCHDOGFILE"
  rm -f "$DB_COMMON"
  cp "$db" "$DB_COMMON"
  sqlite3 "$DB_COMMON" < "$MERGESQL";
  python maps/mapgen/rebalance.py "$VCMIMAP" "$DB_COMMON"
done
