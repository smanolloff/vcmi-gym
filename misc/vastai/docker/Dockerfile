FROM vastai/pytorch:cuda-12.8.1-auto AS builder

WORKDIR /workspace/vcmi-gym

# Check secrets
RUN --mount=type=secret,id=aws_access_key,env=AWS_ACCESS_KEY \
    --mount=type=secret,id=aws_secret_key,env=AWS_SECRET_KEY \
    { [ "${#AWS_ACCESS_KEY}" -eq 20 ] || { echo "AWS_ACCESS_KEY: invalid length: want: 20, have: ${#AWS_ACCESS_KEY}"; exit 1; } } \
    && { [ "${#AWS_SECRET_KEY}" -eq 40 ] || { echo "AWS_SECRET_KEY: invalid length: want: 40, have: ${#AWS_SECRET_KEY}"; exit 1; } }

####################################################
# TODO: on rent, set these as instance ENV
#   - VAST_API_KEY
#   - WANDB_API_KEY
#   - AWS_ACCESS_KEY
#   - AWS_SECRET_KEY
#   - VCMI_ARCHIVE_KEY
#   - VASTAI_INSTANCE_ID
#
# TODO: on rent, exec:
#   sed -e "s/__VAST_API_KEY__/$VAST_API_KEY/" \
#       -e "s/__WANDB_API_KEY__/$WANDB_API_KEY/" \
#       -e "s/__AWS_ACCESS_KEY__/$AWS_ACCESS_KEY/" \
#       -e "s/__AWS_SECRET_KEY__/$AWS_SECRET_KEY/" \
#       -e "s/__VCMI_ARCHIVE_KEY__/$VCMI_ARCHIVE_KEY/" \
#       -i /root/.simorc /root/.aws/credentials
#
#   7z x vcmi/h3.7z -p"$VCMI_ARCHIVE_KEY" -o"vcmi"
#   wandb init -p vcmi-gym && wandb login "$WANDB_API_KEY"
#
#   # This can't be done build-time (tmux throws errors)
#   git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm \
#     && tmux source /root/.tmux.conf \
#     && ~/.tmux/plugins/tpm/bin/install_plugins
#
#   /opt/instance-tools/bin/vastai label instance $VASTAI_INSTANCE_ID ready
#
####################################################

# Everything installed after the first newline are packages from vcmi/CI/before_install/linux_qt5.sh
COPY ./misc/vastai/docker/etc/apt/sources.list /etc/apt/
RUN apt-get update && apt-get -y install \
    7zip tmux vim ccache g++ liblzma-dev libboost-date-time-dev \
    libboost-dev libboost-filesystem-dev libboost-system-dev libboost-thread-dev \
    libboost-program-options-dev libboost-locale-dev libboost-iostreams-dev \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    qtbase5-dev qtbase5-dev-tools qttools5-dev qttools5-dev-tools \
    libqt5svg5-dev \
    ninja-build zlib1g-dev libavformat-dev libswscale-dev libtbb-dev \
    libluajit-5.1-dev libminizip-dev libfuzzylite-dev libsqlite3-dev \
    libsquish-dev

# aws CLI setup
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install --update

RUN --mount=type=secret,id=aws_access_key,env=AWS_ACCESS_KEY \
    --mount=type=secret,id=aws_secret_key,env=AWS_SECRET_KEY \
    mkdir -p /root/.aws \
    && echo "[default]" > /root/.aws/credentials \
    && echo "aws_access_key_id=${AWS_ACCESS_KEY}" >> /root/.aws/credentials \
    && echo "aws_secret_access_key=${AWS_SECRET_KEY}" >> /root/.aws/credentials \
    && chmod 600 /root/.aws/credentials \
    && aws s3 cp s3://vcmi-gym/h3.7z vcmi/

# ONNX setup (copied from vcmi/CI/before_install/linux_common.sh)
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime \
    ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz

RUN mkdir -p "$ONNXRUNTIME_ROOT" \
    && curl -fsSL "$ONNXRUNTIME_URL" | tar -xzv --strip-components=1 -C "$ONNXRUNTIME_ROOT"

COPY ./misc/vastai/docker/etc/cron.d/cleanup /etc/cron.d/
COPY ./misc/vastai/docker/.aws/ /root/.aws/
COPY ./misc/vastai/docker/.vimrc \
    ./misc/vastai/docker/.tmux.conf \
    ./misc/vastai/docker/.gitconfig \
    ./misc/vastai/docker/.gitignore_global \
    ./misc/vastai/docker/runcmd.sh \
    ./misc/vastai/docker/cleanup.sh \
    ./misc/vastai/docker/.simorc \
    /root/

# Use a separate simorc which is called explicitly from nonlogin shells
RUN echo "source ~/.simorc" >> /root/.bashrc \
    && echo 'export VASTAI_INSTANCE_ID=$(cat ~/.vast_containerlabel | cut -c3-)' >> /root/.bashrc \
    && chmod +x /root/cleanup.sh \
    && chmod 644 /etc/cron.d/cleanup # TODO: check if cron is started on boot

COPY . .
RUN . /venv/main/bin/activate \
    && cd vcmi-gym \
    && make pip-install \
    && cd vcmi \
    && mkdir -p data/config \
    && cp ML/configs/*.json data/config \
    && ln -s ../../../maps/gym data/Maps/ \
    && make vastai-build \
    && cd .. \
    && make vastai-build-connector \
    && pip cache purge
