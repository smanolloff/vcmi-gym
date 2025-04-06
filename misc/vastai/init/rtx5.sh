set -x

cat <<-EOF >>~/.bashrc
alias gs='git status'
alias gl='git log'
alias gh='git h'
alias ghh='git hh'
alias ga='git add'
alias gc='git commit'
alias gci='git commit'
alias gd='git diff'
alias gco='git checkout'
export AWS_ACCESS_KEY="AKIATMR________NAX7XH"
export AWS_SECRET_KEY="7d194BC_________________________Opi5dcDA"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $WORKSPACE/vcmi-gym
EOF
# XXX: sourcing ~/.bashrc does not work without interactive terminal
export AWS_ACCESS_KEY="AKIATMR________NAX7XH"
export AWS_SECRET_KEY="7d194BC_________________________Opi5dcDA"

apt -y install vim

git clone --branch dev --single-branch https://github.com/smanolloff/vcmi-gym.git
cd $WORKSPACE/vcmi-gym

sed -i 's/^torch/#torch/' requirements.txt
pip install --pre torch==2.8.0.dev20250315+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
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
tmux source ~/.tmux.conf || :
~/.tmux/plugins/tpm/bin/install_plugins

touch ~/.no_auto_tmux

#
# GIT setup
# use git.sh (would exceed vastai init maxlen)
#

#
# VCMI setup
# use vcmi.sh (would exceed vastai init maxlen)
#

###
### After login (manual):
###
# wandb init -p vcmi-gym && wandb login
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
