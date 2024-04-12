#!/bin/zsh
# XXX: Not using fish due to lack of "set -e" equivalent

USAGE="
Usage:

zsh wandb-upload.zsh data/group_id/run_id/agent_file.pt
zsh wandb-upload.zsh other_dir/agent.pt sp6iid0m
"

set -eux

path=$1

if [ "${2-}" == "" ]; then
    run=$2
else
    # Infer run ID from given path: data/<group>/<run>/<agent>.pt
    run=${$2#*/*/}  # => <run>/<agent>.pt
    run=${run%%/*}  # => <run>
fi

[[ "$run" =~ '^[A-Za-z0-9][A-Za-z0-9_-]+[A-Za-z0-9]$' ]] || { echo "Run $run does not look like an ID"; exit 1; }

wandb artifact put --type model --id "$run" --name "vcmi-gym/model-$run.pt" --description "$path" "$path"
