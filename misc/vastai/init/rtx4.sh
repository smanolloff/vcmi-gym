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
export RAY_memory_monitor_refresh_ms=0
export VASTAI=1
alias python=python3
cd $WORKSPACE/vcmi-gym
set -x
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
tmux source ~/.tmux.conf || :
~/.tmux/plugins/tpm/bin/install_plugins

touch ~/.no_auto_tmux

# APT setup (apt fails to resolve VCMI deps from some of the non-official mirrors)
cat <<-EOF >/etc/apt/sources.list
deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu jammy-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse
EOF

#
# POST-INSTALL: copy-paste code directly in terminal:
# post-install.sh
#

###
### OTHER
###
### To stop the VM from within itself
### (CONTAINER_ID from the UI in the top-left corner of the instance widget)
###
# vastai stop instance <CONTAINER_ID>
###
###
### To log tmux pane in .tmux/:
###
###
# CTRL+B - SHIFT+P
