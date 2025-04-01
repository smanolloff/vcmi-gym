set -x

cat <<-EOF >>~/.bashrc
export AWS_ACCESS_KEY="AKIATMR________NAX7XH"
export AWS_SECRET_KEY="7d194BC________________________Opi5dcDA"
cd $WORKSPACE/vcmi-gym
EOF
# XXX: sourcing ~/.bashrc does not work without interactive terminal
export AWS_ACCESS_KEY="AKIATMR________NAX7XH"
export AWS_SECRET_KEY="7d194BC_________________________Opi5dcDA"

apt -y install vim

git clone --branch dev --single-branch https://github.com/smanolloff/vcmi-gym.git

cd $WORKSPACE/vcmi-gym
pip install -r requirements.txt
pip install vastai

#
# TMUX setup
# (ctrl+b - shift+p) to enable logging
#

git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

cat <<-EOF >~/.tmux.conf
set-option -g history-limit 100000
set-window-option -g mode-keys vi

set -g @logging-path ".tmux"

# Plugins
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-logging'
run '~/.tmux/plugins/tpm/tpm'
EOF

# Install tmux plugins
~/.tmux/plugins/tpm/bin/install_plugins

touch ~/.no_auto_tmux

###
### After login (manual):
###
# wandb init -p vcmi-gym
# python3 -m rl.t10n.util.s3downloader data/t10n/.cache

###
### To stop the VM from within itself
### (CONTAINER_ID from the UI in the top-left corner of the instance widget)
###
# vastai stop instance <CONTAINER_ID>

###
### To log tmux pane in .tmux/:
###
###
# CTRL+B - SHIFT+P
