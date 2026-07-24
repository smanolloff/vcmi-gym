#!/bin/bash

set -euxo pipefail

################
# Faketime
################

# Calculate time drift and optionally sets tmux faketime alias
# HTTP headers end with \r => grep printable chars
ref_date=$(curl -fsSI https://www.google.com | grep -Eio '^date:[[:print:]]+' | cut -d' ' -f2-)
ref_epoch=$(date -u -d "$ref_date" "+%s")
now_epoch=$(date +%s)

# There will always be some small diff (network delay) => trunc to minutes
diffmins=$(((ref_epoch - now_epoch) / 60))

# Add explicit "+"
[ $diffmins -gt 0 ] && offset="+${diffmins}m" || offset="${diffmins}m"

if [ "$offset" !=  "0m" ]; then
    apt-get -o Acquire::Check-Date=false update
    apt-get -o Acquire::Check-Date=false -y install faketime
    faketime_so=$(dpkg -L libfaketime | grep libfaketime.so)

    export FAKETIME=$offset
    export LD_PRELOAD=$faketime_so
    export FAKETIME_DISABLE_SHM=1

    echo "FAKETIME=$offset" >> /etc/environment
    echo "LD_PRELOAD=$faketime_so" >> /etc/environment
    echo "FAKETIME_DISABLE_SHM=1" >> /etc/environment

    # Permanently enable faketime (effective globally and immediately)
    # XXX: this eems to break vastai's key exchange, do not use
    # echo "$faketime_so" > /etc/ld.so.preload
fi

################
### TMUX
################

touch ~/.no_auto_tmux
tmux source ~/.tmux.conf || :
[ -d ~/.tmux/plugins/tpm ] || git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
~/.tmux/plugins/tpm/bin/install_plugins

################
### Cron
################
echo '0 0 * * * root /workspace/vcmi-gym/misc/vastai/docker/cleanup.sh 48 /workspace/vcmi-gym/data/v15 >> /root/cleanup.log' > /etc/cron.d/cleanup
service cron reload

################
### AWS CLI
################

aws configure set aws_access_key_id "$AWS_ACCESS_KEY"
aws configure set aws_secret_access_key "$AWS_SECRET_KEY"

################
### H3 data
################

cd /workspace/vcmi-gym
7z x vcmi/h3.7z -y -p"$VCMI_ARCHIVE_KEY" -o"vcmi/"
/opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID ready

################
### W&B
################

wandb init -p vcmi-gym && wandb login "$WANDB_API_KEY"

################
### Git update
################

if [ "${VASTAI_INIT_UPDATE:-}" = "1" ]; then
    if git fetch --quiet && git merge-base --is-ancestor '@{u}' HEAD; then
        echo "vcmi-gym is up to date"
    else
        echo "vcmi-gym is NOT up to date"
        git pull --recurse-submodules
        make vastai-build
        make vastai-build-connector
    fi
fi

################
### Perf check
################

if [ -n "${VASTAI_INIT_CHECK_ARGS:-}" ]; then
    bash misc/vastai/docker/check.sh $VASTAI_INIT_CHECK_ARGS
fi

################
### Autorun
################

if [ -n "${VASTAI_INIT_AUTORUN_ARGS:-}" ]; then
    bash misc/vastai/docker/autorun.sh $VASTAI_INIT_AUTORUN_ARGS
fi
