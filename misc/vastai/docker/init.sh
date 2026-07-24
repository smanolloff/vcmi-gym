#!/bin/bash

set -euxo pipefail

[ -f /workspace/.initialized ] && exit 0 || :

### TMUX
touch ~/.no_auto_tmux
tmux source ~/.tmux.conf || :
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
~/.tmux/plugins/tpm/bin/install_plugins

### H3 data
7z x vcmi/h3.7z -y -p"$VCMI_ARCHIVE_KEY" -o"vcmi/"
/opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID ready

### W&B
wandb init -p vcmi-gym && wandb login "$WANDB_API_KEY"

touch /workspace/.initialized
