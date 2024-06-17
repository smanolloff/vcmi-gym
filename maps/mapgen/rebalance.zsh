#!/usr/bin/zsh

# Script for evaluating map and optimizing a map
# in cycles of 100K battles (StupidAI vs StupidAI)

# add timestamps to xtrace output
export PS4="+[%D{%Y-%m-%d %H:%M:%S}]<$$> (rebalance.zsh) "

set -eux
map=$1
watchdogfile=$2

[ -r "$map" ] || { echo "Bad map: $map"; exit 1; }

map=$(realpath "$map")
vcmimap=${map#$VCMIGYM/maps/}

[ "$map" = "$VCMIGYM/maps/$vcmimap" ] || { echo "script error"; exit 1; }

db="$(dirname "$map")/$(basename "$map" .vmap).sqlite3"

while true; do
  rm -f "$db"
  for i in $(seq 10); do
    touch "$watchdogfile"
    $VCMI/rel/bin/gymclient-headless \
      --loglevel-ai error --loglevel-global error --loglevel-stats info \
      --random-heroes 1 --random-obstacles 1 --swap-sides 0 \
      --red-ai StupidAI --blue-ai StupidAI \
      --stats-mode red --stats-persist-freq 1000 --stats-sampling 0 \
      --stats-storage "$db" \
      --max-battles 10000 \
      --map "$vcmimap"
  done

  python maps/mapgen/rebalance.py "$map" "$db"
done

