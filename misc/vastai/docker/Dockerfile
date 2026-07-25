FROM vastai/pytorch:cuda-12.8.1-auto

WORKDIR /workspace/vcmi-gym

# Check secrets
# Everything installed after the first newline are packages from vcmi/CI/before_install/linux_qt5.sh
RUN apt-get update && apt-get -y install \
    7zip tmux vim ccache g++ liblzma-dev libboost-date-time-dev \
    libboost-dev libboost-filesystem-dev libboost-system-dev libboost-thread-dev \
    libboost-program-options-dev libboost-locale-dev libboost-iostreams-dev \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    qtbase5-dev qtbase5-dev-tools qttools5-dev qttools5-dev-tools libqt5svg5-dev \
    ninja-build zlib1g-dev libavformat-dev libswscale-dev libtbb-dev \
    libluajit-5.1-dev libminizip-dev libfuzzylite-dev libsqlite3-dev libsquish-dev

# ONNX setup (copied from vcmi/CI/before_install/linux_common.sh)
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime \
    ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VASTAI=1

COPY . .

# Benchmark model
RUN --mount=type=secret,id=aws_access_key,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws_secret_key,env=AWS_SECRET_ACCESS_KEY \
    set -e \
    && ln -s /workspace/vcmi-gym/misc/vastai/docker/.vimrc /root/ \
    && ln -s /workspace/vcmi-gym/misc/vastai/docker/.tmux.conf /root/ \
    && ln -s /workspace/vcmi-gym/misc/vastai/docker/.gitconfig /root/ \
    && ln -s /workspace/vcmi-gym/misc/vastai/docker/.gitignore_global /root/ \
    && ln -s /workspace/vcmi-gym/misc/vastai/docker/.simorc /root/ \
    && cp /workspace/vcmi-gym/misc/vastai/docker/onstart.sh /root/ \
    && mkdir -p data/v15 \
    && mkdir -p "$ONNXRUNTIME_ROOT" \
    && curl -fsSL "$ONNXRUNTIME_URL" | tar -xzv --strip-components=1 -C "$ONNXRUNTIME_ROOT" \
    && for suffix in config.json state-default.json model-ppo.pt optimizer-default.pt scaler-default.pt; do \
        aws s3 cp s3://vcmi-gym/v15/models/pdpyqkrb-benchmark-$suffix data/v15/; \
        ln -s pdpyqkrb-benchmark-$suffix data/v15/pdpyqkrb-$suffix; \
    done \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip -d /root \
    && rm awscliv2.zip \
    && /root/aws/install --update \
    && aws configure set region eu-north-1 \
    && aws configure set output json \
    && aws s3 cp s3://vcmi-gym/h3.7z vcmi/ \
    && echo "source ~/.simorc" >> /root/.bashrc \
    && . /venv/main/bin/activate \
    && make pip-install \
    && cd vcmi \
    && mkdir -p data/config \
    && cp ML/configs/*.json data/config \
    && mkdir -p data/Maps \
    && ln -s ../../../maps/gym data/Maps/ \
    && make vastai-build \
    && cd .. \
    && make vastai-build-connector \
    && pip cache purge

#
# NOTE:
# The VastAI's entrypoint is /opt/instance-tools/bin/entrypoint.sh
#
#   $ docker buildx imagetools inspect ghcr.io/smanolloff/vcmi-gym:latest --format '{{json .Image.Config.Entrypoint}}'
#   [
#     "/opt/instance-tools/bin/entrypoint.sh"
#   ]
#
# However, this entrypoint is NOT used: vast uses cmdline on container create
#
#   # from container itself
#   $ tr '\0' ' ' < /proc/1/cmdline
#   bash /.launch
#
# => the de-facto entrypoint is /.launch
# It installs essentials such as openssh-server, tmux, rsync, etc.
# It also has a bug where it checks and creates .onstart.sh using a relative instead of absolute path,
# resulting in a redundant /workspace/vcmi-gym/onstart.sh file created
#
