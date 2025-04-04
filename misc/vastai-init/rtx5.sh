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

cat <<-EOF >~/.gitconfig
[core]
    pager = less -R
    editor = vim
    excludesfile = ~/.gitignore_global
[color]
    ui = true
[url "https://github.com/"]
    insteadOf = git@github.com:
[alias]
    co = checkout
    ci = commit
    st = status
    br = branch
    d = diff
    h = log --pretty=format:\\"%C(magenta)%h%C(reset) %C(green)%ad%C(reset) %s%d %C(blue)[%an]%C(reset)\\" --graph --date=short -15
    hh = log --pretty=format:\\"%C(magenta)%h%C(reset) %C(green)%ad%C(reset) %s%d %C(blue)[%an]%C(reset)\\" --graph --date=short -15 --first-parent
EOF

cat <<-EOF >~/.gitignore_global
*.swp
*.log
*.orig
*.pyc
.venv
EOF

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

##
## VCMI
##
cat <<-EOF >vcmi.sh
set -eux

[ -n "\$1" ] || { echo "Usage: \$0 H3_ZIP_PASSWORD"; exit 1; }

git submodule update --init --recursive
cd vcmi

cat <<-PYEOF | python3
import os
import boto3
s3 = boto3.client("s3", aws_access_key_id=os.environ["AWS_ACCESS_KEY"], aws_secret_access_key=os.environ["AWS_SECRET_KEY"], region_name="eu-north-1")
s3.download_file("vcmi-gym", "h3.tar.zip", "h3.tar.zip")
PYEOF

unzip -P "\$1" h3.tar.zip
tar -xf h3.tar
rm h3.tar*
mkdir -p data/config
cp ML/configs/*.json data/config

apt update
apt -y install cmake g++ libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev \
    libsdl2-mixer-dev zlib1g-dev libavformat-dev libswscale-dev libboost-dev \
    libboost-filesystem-dev libboost-system-dev libboost-thread-dev libboost-program-options-dev \
    libboost-locale-dev qtbase5-dev libtbb-dev libluajit-5.1-dev qttools5-dev \
    libsqlite3-dev liblzma-dev pybind11-dev python3.10-dev ccache
cmake -S . -B rel -Wno-dev \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=0 \
    -D ENABLE_CCACHE=1 \
    -D ENABLE_NULLKILLER_AI=0 \
    -D ENABLE_LAUNCHER=0 \
    -D ENABLE_ML=1 \
    -D ENABLE_MMAI=1 \
    -D MMAI_LIBTORCH_PATH=""
cmake --build rel/ -- -j16

cd "../vcmi_gym/connectors"
apt -y install libboost-all-dev
cmake -S . -B rel -Wno-dev \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=0
cmake --build rel/ -- -j8

EOF

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

##
## To build vcmi:
##
# bash vcmi.sh H3_ZIP_PASSWORD
