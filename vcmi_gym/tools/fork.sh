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
| .run.resumed_config = "data/mppo-dna-heads/" + $id + "-config.json"
| {
    "user_timeout": 2400,
    "vcmi_timeout": 2400,
    "boot_timeout": 2400
} as $timeouts
| .train.env.kwargs += $timeouts
| .eval.env_variants["BattleAI.open"].kwargs += $timeouts

#
# SPECIFIC TO RUN
#

| {
  "reward_step_fixed": -0.002,
  "reward_term_mult": 0.03,
  "reward_prog_base": 0.1,
  "reward_prog_trigger": 15,
  "reward_prog_exponent": 2,
  "reward_prog_limit": 10,
} as $rewards

| .eval.env_variants["BattleAI.open"].kwargs += $rewards
| .train.env.kwargs += $rewards
| .train.gamma = 0.98
| .model.legacy_global_encoder = false
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
