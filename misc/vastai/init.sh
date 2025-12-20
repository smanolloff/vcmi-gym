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

CPU_COUNT=$1

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

[ -d vcmi-gym ] || git clone --single-branch https://github.com/smanolloff/vcmi-gym.git
cd vcmi-gym

apt-get update
apt-get -y install python3-venv

# An alias will not work in non-login shells
[ -e /usr/bin/python ] || ln -s python3 /usr/bin/python
[ -d .venv ] || python3 -m venv .venv
set +x
. /workspace/vcmi-gym/.venv/bin/activate
set -x

pip install -r requirements.txt
# pip install --break-system-packages jax[cuda12]

#
# ~/.bashrc setup
#

grep ". ~/.simorc" ~/.bashrc || cat <<-EOF >>~/.bashrc
# Need a separate simorc (called explicitly from nonlogin shells)
. ~/.simorc
EOF

cat <<-EOF >~/.simorc
set -x

function pymod() {
    local f=\$1
    shift
    python -m \$(echo \${f//\//.} | sed 's/\.py$//') "\${@}"
}

#
# Link a tagged checkpoint
#
function link_checkpoint() {
    [ -n "\${1:-}" ] || { echo "Usage: link_checkpoint TIMESTAMP [DIR]"; return 1; }
    [ -n "\$2" ] && link_dir="\${2%/}" || link_dir=.

    [ \$1 = best ] || [[ \$1 =~ ^[0-9]+\$ ]] || { echo "Invalid tag: \$1"; return 1; }

    confirmed=no

    for f in \$link_dir/*-\$1-*; do
        fbase=\${f##*/}
        flink=\$link_dir/\${fbase:0:8}-\${fbase:20}
        if [ -e \$flink -a \$confirmed != yes ]; then
            echo -n "File exists, type 'yes' to continue: "
            read -r confirmed
            [ "\$confirmed" = "yes" ] || return 0
        fi

        abstarget="\$link_dir/\$fbase"
        [ -e "\$abstarget" ] || { echo "Error: target not found: \$abstarget"; return 1; }

        ln -fs \$fbase \$link_dir/\${fbase:0:8}-\${fbase:20}
    done
}

#
# Upload a timestamped checkpoint (from current dir)
#
function upload_checkpoint() {
    [ -n "\${1:-}" ] || { echo "Usage: upload_checkpoint RUN_ID-TAG"; return 1; }

    rid=\${1%-*}
    tag=\${1#*-}
    s3_dir=mppo-dna-heads/models

    [ -n "\$rid" -a -n "\$tag" ] || { echo "Usage: upload_checkpoint RUN_ID-TAG"; return 1; }

    files=()
    for suffix in config.json state-default.json model-dna.pt optimizer-distill.pt optimizer-policy.pt optimizer-value.pt scaler-default.pt; do
        f=\$rid-\$tag-\$suffix
        [ -r \$f ] || { echo "Not found: \$f"; return 1; }
        files+=(\$f)
    done

    for f in \${files[@]}; do
        aws s3 cp \$f s3://vcmi-gym/\$s3_dir/
    done
}

#
# Download a timestamped checkpoint (to out_dir as per the config)
#
function download_checkpoint() {
    [ -n "\${1:-}" ] || { echo "Usage: download_checkpoint RUN_ID-TAG"; return 1; }

    rid=\${1%-*}
    tag=\${1#*-}
    s3_dir=mppo-dna-heads/models

    [ -n "\$rid" -a -n "\$tag" ] || { echo "Usage: download_checkpoint RUN_ID-TAG"; return 1; }

    cfg_json="\$(aws s3 cp s3://vcmi-gym/\$s3_dir/\$rid-\$tag-config.json -)"
    [ -n "\$cfg_json" ] || { echo "Failed to fetch config.json"; return 1; }
    out_dir=\$(echo "\$cfg_json" | jq -r '.run.out_dir')
    mkdir -p "\$out_dir"

    # Copy config separately (already downloaded as text)
    echo "\$cfg_json" > \$out_dir/\$rid-\$tag-config.json

    files=()
    for suffix in state-default.json model-dna.pt optimizer-distill.pt optimizer-policy.pt optimizer-value.pt scaler-default.pt; do
        files+=(\$rid-\$tag-\$suffix)
    done

    for f in \${files[@]}; do
      aws s3 cp s3://vcmi-gym/\$s3_dir/\$f \$out_dir/  || { echo "ERROR"; return 1; }
    done
}

#
# Make a timestamped checkpoint
#
function backup_best_checkpoint() {
    [ -n "\${1:-}" ] || { echo "Usage: backup_best_checkpoint ID [TIMESTAMP]"; return 1; }
    [ -n "\${2:-}" ] && ts=\$2 || ts="\$(date +%s)"

    id=\$1

    [[ \$ts =~ ^[0-9]+\$ ]] || { echo "Invalid timestamp: \$ts"; return 1; }
    [[ \$id =~ ^[a-z]{8}\$ ]] || { echo "Bad id: \$id"; return 1; }

    if [ -e \$id-\$ts-model-dna.pt ]; then
        echo "File already exists: \$id-\$ts-model-dna.pt"
        return 1
    fi

    for f in config state-default; do
      cp \$id-{best-,\$ts-}\$f.json
    done

    for f in model-dna optimizer-distill optimizer-policy optimizer-value scaler-default; do
      cp \$id-{best-,\$ts-}\$f.pt
    done
}

function setboot() {
    cat <<EOL >/root/onstart.sh
#!/bin/bash
# Logs are in /var/log/onstart.log

set -x

# For tracking reboots
date +'BOOTED_AT=%FT%T%z' >> ~/.bashrc

tmux new-session -d "source ~/.simorc; cd \$PWD; \$*; exec \$SHELL"
EOL
}

function retry_until_sigint() {
    local i=0
    while true; do
        bash -x ~/runcmd.sh \$*
        if [ \$? -eq 130 ]; then
            echo "Interrupt detected, will NOT retry"
            /opt/instance-tools/bin/vastai label instance \$VASTAI_INSTANCE_ID IDLE
            break
        else
            echo "Interrupt NOT detected, will retry in 10s..."
            let ++i
            /opt/instance-tools/bin/vastai label instance \$VASTAI_INSTANCE_ID CRASHED_\$i
            # FIXME: if there are other runs on the same instance, this will kill them as well
            killall python
            sleep 10
        fi
    done
}

function train_gnn() {
    # To start a new run, pass "" explicitly, e.g.
    [ "\$#" -eq 1 ] || [ "\$#" -eq 2 -a "\$2" = "--dry-run" ] || {
        cat <<-USAGE
Usage: train_gnn run_id [--dry-run]
USAGE
        return 1
    }

    if [ -z "\$1" ]; then
        LC_ALL=C run_id=\$(tr -dc 'a-z' </dev/urandom | head -c8)
    else
        run_id="\$1"
    fi

    f="data/mppo-dna-heads/\$run_id-config.json"

    basecmd="python -m rl.algos.mppo_dna_gnn.mppo_dna_gnn"

    if [ -z "\$2" ]; then
        setboot retry_until_sigint \$basecmd -f \$f
        extra_args=
    else
        # --dry-run => don't resume on boot (let it transition to IDLE)
        setboot :
        basecmd+=" --dry-run"
    fi

    if ! [ -r "\$f" ]; then
        echo "This is a new run -- will start ONCE with --run-id to create it"
        bash -x ~/runcmd.sh \$basecmd --run-id \$run_id

        if [ \$? -eq 130 ]; then
            echo "Interrupt detected, will NOT retry"
            /opt/instance-tools/bin/vastai label instance \$VASTAI_INSTANCE_ID IDLE
            # run was interrupted, no need to resume on boot
            setboot :
            return 0
        fi

        # The config file should now be created, can enter the regular retry loop
        echo "Interrupt NOT detected, will enter retry loop"
        /opt/instance-tools/bin/vastai label instance \$VASTAI_INSTANCE_ID CRASHED
        killall python
        sleep 10
    fi

    retry_until_sigint \$basecmd -f \$f
    setboot :
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
set +x
. /workspace/vcmi-gym/.venv/bin/activate
set -x
EOF

cat <<-EOF >~/runcmd.sh
#!/bin/bash
trap 'echo "INTERRUPT DETECTED, EXIT 130"; exit 130' INT

df  # for logging
mb=\$(df --output=avail -m / | tail -1 | awk '{print \$1}')
if ! [[ \$mb =~ ^[0-9]+\$ ]]; then
    echo "Failed to determine free space: '\$mb'"
    exit 1
fi

if [ \$mb -lt 500 ]; then
    echo "Less than 500MB of free space: '\$mb'"
    exit 1
fi

set +x
. /workspace/vcmi-gym/.venv/bin/activate
set -x

# XXX: do NOT use exec here (does not call trap)
\$@

echo "PROGRAM EXIT CODE: \$?"
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

if ! [ -f h3.tar ]; then
    cat <<-PYEOF | python3
import os
import boto3
s3 = boto3.client("s3", aws_access_key_id=os.environ["AWS_ACCESS_KEY"], aws_secret_access_key=os.environ["AWS_SECRET_KEY"], region_name="eu-north-1")
s3.download_file("vcmi-gym", "h3.tar.zip", "h3.tar.zip")
PYEOF

    unzip -P "$VCMI_ARCHIVE_KEY" h3.tar.zip
fi
rm -rf data
tar -xf h3.tar
mkdir -p data/config
cp ML/configs/*.json data/config
ln -s ../../../maps/gym data/Maps/

if ! [ -d /opt/onnxruntime/lib/libonnxruntime.so ]; then
    # Copied from vcmi/CI/before_install/linux_common.sh
    ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz
    ONNXRUNTIME_ROOT=/opt/onnxruntime
    mkdir -p "$ONNXRUNTIME_ROOT"
    curl -fsSL "$ONNXRUNTIME_URL" | tar -xzv --strip-components=1 -C "$ONNXRUNTIME_ROOT"
fi

# Everything after the first newline are packages from vcmi/CI/before_install/linux_qt5.sh
apt-get -y install vim ccache cmake g++ liblzma-dev \
  libboost-dev libboost-filesystem-dev libboost-system-dev libboost-thread-dev \
  libboost-program-options-dev libboost-locale-dev libboost-iostreams-dev \
  libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
  qtbase5-dev qtbase5-dev-tools qttools5-dev qttools5-dev-tools \
  libqt5svg5-dev \
  ninja-build zlib1g-dev libavformat-dev libswscale-dev libtbb-dev \
  libluajit-5.1-dev libminizip-dev libfuzzylite-dev libsqlite3-dev \
  libsquish-dev

cmake -S . -B rel -Wno-dev \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=0 \
    -D ENABLE_CCACHE=1 \
    -D ENABLE_NULLKILLER_AI=0 \
    -D ENABLE_LAUNCHER=0 \
    -D ENABLE_ML=1 \
    -D ENABLE_MMAI=1 \
    -D MMAI_EXECUTORCH_PATH=""
cmake --build rel/ -- -j$CPU_COUNT

cd ../vcmi_gym/connectors
apt-get -y install libboost-all-dev
cmake -S . -B rel -Wno-dev \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=0
cmake --build rel/ -- -j$CPU_COUNT

cd ../../
wandb init -p vcmi-gym && wandb login "$WANDB_API_KEY"

pip cache purge

finished_at=$(date +%s)

echo "Done in $((finished_at - started_at)) seconds."

# Mark as completed
touch /workspace/.initialized
/opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID ready
