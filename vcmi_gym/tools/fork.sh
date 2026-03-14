#!/bin/bash

# Usage:
#
#   fork-run.sh tukbajrv-202509241418 tukba-siege-g98
#
# WARNING:
# - works only on VastAI instances
#

source ~/.simorc

set -euxo pipefail

SRC_ID=${1?SRC_ID required as \$1}
SUFFIX=${2?SUFFIX required as \$2}

data=data/mppo-dna-heads

if ! [ -e $data/$SRC_ID-config.json -a -e $data/$SRC_ID-model-dna.pt ]; then
    download_checkpoint $SRC_ID
fi

# subshell exits with 141 (SIGPIPE) when -o pipefail is set => use :
LC_ALL=C id=$(tr -dc 'a-z' </dev/urandom | head -c8) || :
prefix=$(date "+%Y%m%d_%H%M%S")
name=$prefix-$id-$SUFFIX

copy_checkpoint -y $SRC_ID $id-fork $data/
link_checkpoint -y $id-fork $id $data/

# cat $data/$SRC_ID-config.json \
# | jq --arg name "$name" \

cat <<'JQ' >$data/$id-fork.jq
#
# Input args:
#   $id     run id, e.g. "zoacpwry"
#   $name   run name, e.g. "20260314_211800-zoacpwry-fork-mmai"
#
.name_template = "{datetime}-{id}-" + $id
| .run.id = $id
| .run.name = $name
| .run.resumed_config = "$data/" + $id + "-config.json"
| {
    "user_timeout": 2400,
    "vcmi_timeout": 2400,
    "boot_timeout": 2400
} as $timeouts
| .train.env.kwargs += $timeouts
| .eval.env_variants["BattleAI.open"].kwargs += $timeouts
| .eval.env_variants["BattleAI.town"].kwargs += $timeouts

#
# SPECIFIC TO RUN
#

| {
    "type": "static",
    "config_file": "data/mppo-dna-heads/nkjrmrsq-202509291846-config.json",
    "weights_file": "data/mppo-dna-heads/nkjrmrsq-202509291846-model-dna.pt"
} as $model

| .eval.env_variants["BattleAI.open"].num_envs_per_opponent.BattleAI = 10
| .eval.env_variants["BattleAI.town"].num_envs_per_opponent.BattleAI = 10

| .eval.env_variants["MMAI.open"] = .eval.env_variants["BattleAI.open"]
| .eval.env_variants["MMAI.open"].kwargs += $timeouts
| .eval.env_variants["MMAI.open"].model = $model
| .eval.env_variants["MMAI.open"].num_envs_per_opponent.model = 10
| .eval.env_variants["MMAI.open"].num_envs_per_opponent.BattleAI = 0

| .eval.env_variants["MMAI.town"] = .eval.env_variants["BattleAI.town"]
| .eval.env_variants["MMAI.town"].kwargs += $timeouts
| .eval.env_variants["MMAI.town"].model = $model
| .eval.env_variants["MMAI.town"].num_envs_per_opponent.model = 10
| .eval.env_variants["MMAI.town"].num_envs_per_opponent.BattleAI = 0

| .train.env.model = $model
| .train.env.num_envs_per_opponent.model = 30
| .train.env.num_envs_per_opponent.BattleAI = 10
JQ

python -c "
import wandb
wandb.init(
    project='vcmi-gym',
    group='mppo-dna-heads',
    name='$name',
    id='$id',
    resume='never',
    config=dict(_start_infos=[{
        'source': '$SRC_ID',
        'jq': open('$data/$id-fork.jq', 'r').read()
    }]),
    sync_tensorboard=False,
    save_code=False,
)
"

cfgsrc=$data/$SRC_ID-config.json
cfgdst=$data/$id-config.json

jq --arg id "$id" --arg name "$name" -f $data/$id-fork.jq $cfgsrc > $cfgdst

cat <<-EOF

New config: $cfgdst

Done.

    train_gnn ${id}

EOF
