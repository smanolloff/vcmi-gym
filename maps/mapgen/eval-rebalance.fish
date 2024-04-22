#!/usr/bin/env fish

#
# XXX: this script should be exececuted from the vcmi-gym root dir:
#      e.g. `fish maps/mapgen/eval-rebalance.fish`
#

for i in (seq 1 10)
  date +"[%F %T] Starting map-eval -> out-$i.json"
  rm -f out-$i.json

  # `unbuffer` on Mac is provided by the homebrew `expect` formula
  $VCMI/rel/bin/myclient-headless \
    --gymdir $VCMIGYM \
    --map gym/generated/4096/4096-mixstack-100K-02.vmap \
    --loglevel-ai error \
    --loglevel-global error \
    --attacker-ai StupidAI \
    --defender-ai StupidAI \
    --random-combat 1 \
    --map-eval 150000 | grep --line-buffered . > out-$i.json

  date +"[%F %T] Checking out-$i.json..."
  if python maps/mapgen/rebalance.py --dry-run -a -f out-$i.json 2>/dev/null
    echo "OK. Rebalancing..."
    python maps/mapgen/rebalance.py -f out-$i.json
  else
    echo "Garbage. Retrying..."
    continue
  end
end
