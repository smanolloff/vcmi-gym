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
# e.g. link_checkpoint joymfkqj-1767449899 joymfkqj
#
function link_checkpoint() {
    [ -n "\${1:-}" -a -n "\${2:-}" ] || { echo "Usage: link_checkpoint PREFIX LINK_PREFIX [DIR]"; return 1; }
    [ -n "\$3" ] && workdir="\${3%/}" || workdir=.

    prefix=\$1
    lprefix=\$2

    (
        set -e
        cd \$workdir

        if ! [ -e \$prefix-model-dna.pt ]; then
            echo "Source model not found: \$prefix-model-dna.pt"
            return 1
        fi

        confirmed=no
        for f in \$prefix-*; do
            tail=\${f#\$prefix-}
            flink=\$lprefix-\$tail

            if [ -e \$flink -a \$confirmed != yes ]; then
                echo -n "File exists, type 'yes' to continue: "
                read -r confirmed
                [ "\$confirmed" = "yes" ] || return 0
            fi

            ln -fs \$f \$flink
        done
    )
}

#
# Upload a timestamped checkpoint (from current dir)
#
function upload_checkpoint() {
    [ -n "\${1:-}" ] || { echo "Usage: upload_checkpoint RUN_ID-TAG [DIR]"; return 1; }
    [ -n "\$2" ] && workdir="\${2%/}" || workdir=.

    rid=\${1%-*}
    tag=\${1#*-}
    s3_dir=mppo-dna-heads/models

    [ -n "\$rid" -a -n "\$tag" ] || { echo "Usage: upload_checkpoint RUN_ID-TAG"; return 1; }

    (
        set -e
        cd \$workdir

        files=()
        for suffix in config.json state-default.json model-dna.pt optimizer-distill.pt optimizer-policy.pt optimizer-value.pt scaler-default.pt; do
            f=\$rid-\$tag-\$suffix
            [ -r \$f ] || { echo "Not found: \$f"; return 1; }
            files+=(\$f)
        done

        for f in \${files[@]}; do
            aws s3 cp \$f s3://vcmi-gym/\$s3_dir/
        done
    )
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
# Copy a checkpoint
# E.g. fdqwrsd-best-... gdhsgsi-202601011251-...
#
function copy_checkpoint() {
    [ -n "\${1:-}" -a -n "\${2:-}" ] || { echo "Usage: copy_checkpoint RUN_ID-TAG1 RUN_ID-TAG2 [DIR]"; return 1; }
    [ -n "\$3" ] && workdir="\${3%/}" || workdir=.

    rid1=\${1%-*}
    tag1=\${1#*-}
    rid2=\${2%-*}
    tag2=\${2#*-}

    [[ \$rid1 =~ ^[a-z]+$ ]] || { echo "Bad RUN_ID1: \$rid1"; return 1; }
    [[ \$rid2 =~ ^[a-z]+$ ]] || { echo "Bad RUN_ID2: \$rid2"; return 1; }
    [[ \$tag1 =~ ^[0-9a-z]+$ ]] || { echo "Bad TAG1: \$tag1"; return 1; }
    [[ \$tag2 =~ ^[0-9a-z]+$ ]] || { echo "Bad TAG2: \$tag2"; return 1; }

    (
        set -e
        cd \$workdir

        confirmed=no
        if [ \$rid1 != \$rid2 ]; then
            echo -n "RUN_ID is different, type 'yes' to continue: "
            read -r confirmed
            [ "\$confirmed" = "yes" ] || return 0
        fi

        prefix1=\$rid1-\$tag1
        prefix2=\$rid2-\$tag2

        if ! [ -e \$prefix1-model-dna.pt ]; then
            echo "Source model not found: \$prefix1-model-dna.pt"
            return 1
        fi

        if [ -e \$prefix2-model-dna.pt ]; then
            echo "Destination already exists: \$prefix2-model-dna.pt"
            return 1
        fi

        for f in config state-default; do
          cp {\$prefix1-,\$prefix2-}\$f.json
        done

        for f in model-dna optimizer-distill optimizer-policy optimizer-value scaler-default; do
          cp {\$prefix1-,\$prefix2-}\$f.pt
        done
    )
}

function setboot() {
    cat <<EOL >/root/onstart.sh
#!/bin/bash
# Logs are in /var/log/onstart.log

set -x

# For tracking reboots
date +'BOOTED_AT=%FT%T%z' >> ~/.bashrc

service cron start || :

tmux new-session -d "source ~/.simorc; tmux rename-window REBOOTED; /opt/instance-tools/bin/vastai label instance \$VASTAI_INSTANCE_ID REBOOTED; cd \$PWD; \$*; exec \$SHELL"
EOL
}

function retry_until_sigint() {
    local run_id=\$1
    shift
    local i=0
    while true; do
        tmux rename-window \$run_id || :
        bash -x ~/runcmd.sh \$*
        if [ \$? -eq 130 ]; then
            echo "Interrupt detected, will NOT retry"
            /opt/instance-tools/bin/vastai label instance \$VASTAI_INSTANCE_ID IDLE
            tmux rename-window IDLE || :
            break
        else
            echo "Interrupt NOT detected, will retry in 10s..."
            let ++i
            /opt/instance-tools/bin/vastai label instance \$VASTAI_INSTANCE_ID CRASHED_\$i
            tmux rename-window CRASHED_\$i || :
            # FIXME: if there are other runs on the same instance, this will kill them as well
            killall python
            sleep 10
        fi
    done
}

function train_gnn() {
    local run_id=\$1
    shift
    local rest=\$*

    local new_run=false
    if [ -z "\$run_id" ]; then
        LC_ALL=C run_id=\$(tr -dc 'a-z' </dev/urandom | head -c8)
        new_run=true
    fi

    [[ \$run_id =~ ^[a-z]{8}\$ ]] || {
        cat <<-USAGE
Usage: train_gnn run_id [opts]
USAGE
        return 1
    }

    local dry_run
    echo \$* | grep -- --dry-run && dry_run=true || dry_run=false

    local f="data/mppo-dna-heads/\$run_id-config.json"

    if [ -r "\$f" ] && \$new_run; then
        echo "Config already exists, but this should be a new run"
        retirn 1
    fi

    if ! [ -r "\$f" ] && ! \$new_run; then
        echo "Config not found, but this should be a resumed run"
        return 1
    fi

    local basecmd="python -m rl.algos.mppo_dna_gnn.mppo_dna_gnn"
    local newcmd="\$basecmd --run-id \$run_id \$rest"

    # Resuming does not use any args except -f and (optionally) --dry-run
    local resumecmd="\$basecmd -f \$f"
    if \$dry_run; then
        resumecmd+=" --dry-run"

        # don't resume on boot (let it transition to REBOOTED)
        setboot :
    else
        # must not start a new run on boot => use resumecmd
        setboot retry_until_sigint \$run_id \$resumecmd
    fi

    if \$new_run; then
        echo "This is a new run -- will start ONCE with --run-id to create it"
        tmux rename-window \$run_id || :
        bash -x ~/runcmd.sh \$newcmd --skip-eval

        if [ \$? -eq 130 ]; then
            echo "Interrupt detected, will NOT retry"
            tmux rename-window IDLE || :
            /opt/instance-tools/bin/vastai label instance \$VASTAI_INSTANCE_ID IDLE
            # run was interrupted, no need to resume on boot
            setboot :
            return 0
        fi

        # The config file should now be created, can enter the regular retry loop
        echo "Interrupt NOT detected, will enter retry loop"
        tmux rename-window CRASHED || :
        /opt/instance-tools/bin/vastai label instance \$VASTAI_INSTANCE_ID CRASHED
        killall python
        sleep 10
    fi

    retry_until_sigint \$run_id \$resumecmd
    setboot :
}

set +x
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

cat <<-EOFCLEANUP >/root/cleanup.sh
#!/bin/bash

set -euxo pipefail

# For logging
date

# Deletes non-symlinked files older than X hours
# Usage:
#   cleanup_data 12 /workspace/vcmi-gym/data/mppo-dna-heads

HOURS="\${1:?Hours required}"
DIR="\${2:?Directory required}"

cd "\$DIR"
[[ "\$HOURS" =~ ^[0-9]+\$ ]] || { echo "Invalid hours: \$HOURS"; exit 1; }
MINUTES=\$(( HOURS * 60 ))

# Associative array filename => 1
# Contains files which are targets of symlinks
declare -A LINK_TARGETS=()
for link in \$(find . -maxdepth 1 -type l -printf '%f\n' | sort); do
    target=\$(readlink "\$link")
    LINK_TARGETS["\$target"]=1
done

DELETED=()
for file in \$(find . -maxdepth 1 -type f -mmin "+\$MINUTES" -printf '%f\n' | sort); do
    [ "\${LINK_TARGETS[\$file]-}" = "1" ] && continue || :
    rm -f "\$file"
    DELETED+=("\$file")
done

cat <<EOF
========================
Deleted:
\$(printf '%s\n' "\${DELETED[@]}")
EOF
EOFCLEANUP

chmod +x /root/cleanup.sh
cat <<EOF >>/etc/cron.d/cleanup
0 0 * * * root /root/cleanup.sh 24 /workspace/vcmi-gym/data/mppo-dna-heads >> /root/cleanup.log
EOF
chmod 644 /etc/cron.d/cleanup
service cron start

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

ONNXRUNTIME_ROOT=/opt/onnxruntime
if ! [ -d /opt/onnxruntime/lib/libonnxruntime.so ]; then
    # Copied from vcmi/CI/before_install/linux_common.sh
    ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz
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
    -D ONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
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
tmux rename-window ready || :
/opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID ready
