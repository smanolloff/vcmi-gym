#!/bin/bash

set -euxo pipefail

function main() {
    ################################################
    # Env
    ################################################

    # XXX: Custon env vars (wandb keys, aws keys, etc.) are available during boot
    # as the container itself was created with this env.
    # However, they are NOT available in other sessions (e.g. in ssh sessions)
    # => store them in /etc/environment to be auto-loaded by PAM
    #
    # XXX: VASTAI_INSTANCE_ID is a special env var as it is not available during boot.
    #      => it is excplicitly set in .simorc
    source ~/.simorc

    cat <<-EOF >>/etc/environment
AWS_ACCESS_KEY=$AWS_ACCESS_KEY
AWS_SECRET_KEY=$AWS_SECRET_KEY
VCMI_ARCHIVE_KEY=$VCMI_ARCHIVE_KEY
VAST_API_KEY=$VAST_API_KEY
WANDB_API_KEY=$WANDB_API_KEY
VASTAI_INSTANCE_ID=$VASTAI_INSTANCE_ID
EOF

    set_label init...

    ################################################
    # Faketime
    ################################################

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

    ################################################
    ### TMUX
    ################################################

    [ -d ~/.tmux/plugins/tpm ] || git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
    tmux source ~/.tmux.conf  # before install_plugins
    ~/.tmux/plugins/tpm/bin/install_plugins
    touch ~/.no_auto_tmux

    ################################################
    ### Cron
    ################################################
    echo '0 0 * * * root /workspace/vcmi-gym/misc/vastai/docker/cleanup.sh 48 /workspace/vcmi-gym/data/v15 >> /root/cleanup.log' > /etc/cron.d/cleanup
    service cron restart

    ################################################
    ### AWS CLI
    ################################################

    aws configure set aws_access_key_id "$AWS_ACCESS_KEY"
    aws configure set aws_secret_access_key "$AWS_SECRET_KEY"
    chmod 600 ~/.aws/credentials

    ################################################
    ### H3 data
    ################################################

    cd /workspace/vcmi-gym
    7z x vcmi/h3.7z -y -p"$VCMI_ARCHIVE_KEY" -o"vcmi/"

    ################################################
    ### W&B
    ################################################

    wandb init -p vcmi-gym && wandb login "$WANDB_API_KEY"

    ################################################
    ### Perf check
    ################################################

    if [ -n "${VASTAI_INIT_CHECK_ARGS:-}" ]; then
        bash misc/vastai/docker/check.sh $VASTAI_INIT_CHECK_ARGS
    fi

    ################################################
    ### Git update (after perf check)
    ################################################

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
}

if [ -f /workspace/.init ]; then
    exit 0
fi

if main "$@"; then
    set_label IDLE
    tmux rename-window $VASTAI_INSTANCE_ID:IDLE || :
    touch /workspace/.init

    ################################################
    ### Autorun
    ################################################
    if [ -n "${VASTAI_INIT_AUTORUN_ARGS:-}" ]; then
        # autorun manages the instance labels from here on
        bash misc/vastai/docker/autorun.sh $VASTAI_INIT_AUTORUN_ARGS || :
    fi

    exit 0
else
    set_label ERR_INIT
    tmux rename-window $VASTAI_INSTANCE_ID:ERR_INIT || :
    echo "INIT ERROR" >&2
    exit 1
fi
