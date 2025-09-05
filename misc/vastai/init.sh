#!/bin/bash

### HOW TO USE THIS SCRIPT"
### 0. Create misc/vastai/.env (if not already created)
### 1. scp misc/vastai/{.env,.init.sh} root@...
### 2. ssh root@... 'tmux new-session -d "bash -xc \"bash /workspace/init.sh; exec \\$SHELL\""'
###
### This will create run the init script within tmux.


# The first char will be used as a depth level indicator by bash
export PS4='+ [$(date "+%Y-%m-%dT%H:%M:%S%z")] '

set -euxo pipefail

if [ -e /workspace/.initialized ]; then
    echo "Already initialized, nothing to do."
    exit 0
fi

set -x

started_at=$(date +%s)

EFFECTIVE_NUM_CPUS=${1:-12}

#
# TMUX setup
# (ctrl+b - shift+p) to enable logging
#

touch ~/.no_auto_tmux
cat <<-EOF >~/.tmux.conf
set-option -g history-limit 100000
set-window-option -g mode-keys vi
EOF

# This takes some time => make it a separate step
if ! [ -e ~/.tmux/plugins/tpm ]; then
    git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
    cat <<-EOF >>~/.tmux.conf
set -g @logging-path ".tmux"

# Plugins
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-logging'
run '~/.tmux/plugins/tpm/tpm'
EOF
fi

tmux source ~/.tmux.conf || :
~/.tmux/plugins/tpm/bin/install_plugins


set -a
source /workspace/.env
set +a

cd /workspace

export VASTAI_INSTANCE_ID=$(cat ~/.vast_containerlabel | cut -c3-)

if [ "${#AWS_ACCESS_KEY}" -ne 20 ]; then
    echo "Invalid AWS_ACCESS_KEY."
    exit 1
fi

if [ "${#AWS_SECRET_KEY}" -ne 40 ]; then
    echo "Invalid AWS_SECRET_KEY."
    exit 1
fi

if [ "${#VCMI_ARCHIVE_KEY}" -ne 16 ]; then
    echo "Invalid VCMI_ARCHIVE_KEY."
    exit 1
fi

if [ "${#WANDB_API_KEY}" -ne 40 ]; then
    echo "Invalid WANDB_API_KEY."
    exit 1
fi

if ! [[ $VASTAI_INSTANCE_ID =~ ^[[:digit:]]+$ ]]; then
    echo "Invalid VASTAI_INSTANCE_ID."
    exit 1;
fi

/opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID init...

# An alias will not work in non-login shells
[ -e /usr/bin/python ] || ln -s python3 /usr/bin/python

if ! [ -d aws ]; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    ./aws/install --update
    mkdir -p ~/.aws
    cat <<-EOF >~/.aws/credentials
[default]
aws_access_key_id = $AWS_ACCESS_KEY
aws_secret_access_key = $AWS_SECRET_KEY
EOF
    cat <<-EOF >~/.aws/config
[default]
region = eu-north-1
output = json
EOF
fi

git clone --branch dev --single-branch https://github.com/smanolloff/vcmi-gym.git
cd vcmi-gym

pip install --break-system-packages -r requirements.txt
# pip install --break-system-packages jax[cuda12]

#
# ~/.bashrc setup
#

cat <<-EOF >>~/.bashrc

function pymod() {
    local f=\$1
    shift
    python -m \$(echo \${f//\//.} | sed 's/\.py$//') "\${@}"
}

#
# Copy a timestamped checkpoint to a regular checkpoint (in current dir)
#
function copy_checkpoint() {
    [ -n "\${1:-}" ] || { echo "Usage: copy_checkpoint TIMESTAMP"; return 1; }

    for f in *\$1*; do
        cp \$f \${f:0:8}-\${f:20}
    done
}

#
# Upload a timestamped checkpoint (from current dir)
#
function upload_checkpoint() {
    [ -n "\${1:-}" ] || { echo "Usage: upload_checkpoint TIMESTAMP"; return 1; }

    cfg_ary=(./*\$1-config.json)  # must use array for globbing
    cfg="\${cfg_ary[0]}"

    [ -r "\$cfg" ] || { echo "No matches for config file: \$cfg_ary"; return 1; }

    s3_dir=\$(jq -r '.checkpoint.s3.s3_dir' \$cfg)

    for f in *\$1*; do
        aws s3 cp \$f s3://vcmi-gym/\$s3_dir/
    done
}

#
# Download a timestamped checkpoint (to out_dir as per the config)
#
function download_checkpoint() {
    [ -n "\${1:-}" ] || { echo "Usage: download_checkpoint RUN_ID-TIMESTAMP [S3_DIR]"; return 1; }

    rid=\${1%-*}
    ts=\${1#*-}
    [ -n "\$2" ] && s3_dir="\${2%/}" || s3_dir=mppo-dna-heads/models

    cfg_json="\$(aws s3 cp s3://vcmi-gym/\$s3_dir/\$rid-\$ts-config.json -)"
    out_dir=\$(echo "\$cfg_json" | jq -r '.run.out_dir')
    mkdir -p "\$out_dir"

    # Copy json files
    echo "\$cfg_json" > \$out_dir/\$rid-\$ts-config.json
    aws s3 cp s3://vcmi-gym/\$s3_dir/\$rid-\$ts-state-default.json \$out_dir/  || { echo "ERROR"; break; }

    # Copy .pt files
    for f in model-dna optimizer-distill optimizer-policy optimizer-value scaler-default; do
      aws s3 cp s3://vcmi-gym/\$s3_dir/\$rid-\$ts-\$f.pt \$out_dir/  || { echo "ERROR"; break; }
    done
}

alias gs='git status'
alias gl='git log'
alias gh='git h'
alias ghh='git hh'
alias ga='git add'
alias gc='git commit'
alias gci='git commit'
alias gd='git diff'
alias gco='git checkout'
export AWS_ACCESS_KEY="$AWS_ACCESS_KEY"
export AWS_SECRET_KEY="$AWS_SECRET_KEY"
export WANDB_API_KEY='$WANDB_API_KEY'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_memory_monitor_refresh_ms=0
export VASTAI=1
export VASTAI_INSTANCE_ID=$VASTAI_INSTANCE_ID
cd /workspace/vcmi-gym
set -x
EOF

# does not work (fails with unbound variable)
# i.e. better not use this tmux session anything else after init
# source ~/.bashrc

# APT setup (apt fails to resolve VCMI deps from some of the non-official mirrors)
cat <<-EOF >/etc/apt/sources.list
deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu jammy-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse
EOF

#
# git setup
#

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
# VCMI setup
#

git submodule update --init --recursive
cd vcmi

cat <<-PYEOF | python3
import os
import boto3
s3 = boto3.client("s3", aws_access_key_id=os.environ["AWS_ACCESS_KEY"], aws_secret_access_key=os.environ["AWS_SECRET_KEY"], region_name="eu-north-1")
s3.download_file("vcmi-gym", "h3.tar.zip", "h3.tar.zip")
PYEOF

unzip -P "$VCMI_ARCHIVE_KEY" h3.tar.zip
tar -xf h3.tar
rm h3.tar*
mkdir -p data/config
cp ML/configs/*.json data/config
ln -s ../../../maps/gym data/Maps/

apt-get update
apt-get -y install vim cmake g++ libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev \
    libsdl2-mixer-dev zlib1g-dev libavformat-dev libswscale-dev libboost-dev \
    libboost-filesystem-dev libboost-system-dev libboost-thread-dev libboost-program-options-dev \
    libboost-locale-dev qtbase5-dev libtbb-dev libluajit-5.1-dev qttools5-dev \
    libsqlite3-dev liblzma-dev python3-dev ccache

cmake -S . -B rel -Wno-dev \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=0 \
    -D ENABLE_CCACHE=1 \
    -D ENABLE_NULLKILLER_AI=0 \
    -D ENABLE_LAUNCHER=0 \
    -D ENABLE_ML=1 \
    -D ENABLE_MMAI=1 \
    -D MMAI_EXECUTORCH_PATH=""
cmake --build rel/ -- -j32

cd "../vcmi_gym/connectors"
apt-get -y install libboost-all-dev
cmake -S . -B rel -Wno-dev \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=0
cmake --build rel/ -- -j8

cd ../../
wandb init -p vcmi-gym && wandb login "$WANDB_API_KEY"

finished_at=$(date +%s)

echo "Done in $((finished_at - started_at)) seconds."

# Mark as completed
touch /workspace/.initialized
/opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID ready
