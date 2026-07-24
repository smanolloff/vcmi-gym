FROM vastai/pytorch:cuda-12.8.1-auto AS builder

WORKDIR /workspace/vcmi-gym

# Check secrets
RUN --mount=type=secret,id=aws_access_key,env=AWS_ACCESS_KEY \
    --mount=type=secret,id=aws_secret_key,env=AWS_SECRET_KEY \
    { [ "${#AWS_ACCESS_KEY}" -eq 20 ] || { echo "AWS_ACCESS_KEY: invalid length: want: 20, have: ${#AWS_ACCESS_KEY}"; exit 1; } } \
    && { [ "${#AWS_SECRET_KEY}" -eq 40 ] || { echo "AWS_SECRET_KEY: invalid length: want: 40, have: ${#AWS_SECRET_KEY}"; exit 1; } }

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
    ./misc/vastai/docker/.simorc \
    ./misc/vastai/docker/cleanup.sh \
    ./misc/vastai/docker/entrypoint.sh \
    ./misc/vastai/docker/init.sh \
    ./misc/vastai/docker/runcmd.sh \
    /root/

# Use a separate simorc which is called explicitly from nonlogin shells
RUN chmod +x /root/entrypoint.sh \
    && chmod +x /root/cleanup.sh \
    && echo 'export VASTAI_INSTANCE_ID=$(cat ~/.vast_containerlabel | cut -c3-)' >> /root/.bashrc \
    && echo "source ~/.simorc" >> /root/.bashrc \
    && mkdir data/v15

# Benchmark model
RUN --mount=type=secret,id=aws_access_key,env=AWS_ACCESS_KEY \
    --mount=type=secret,id=aws_secret_key,env=AWS_SECRET_KEY \
    for suffix in config.json state-default.json model-ppo.pt optimizer-default.pt scaler-default.pt; do \
        aws s3 cp s3://vcmi-gym/v15/models/pdpyqkrb-benchmark-$suffix data/v15/; \
        ln -s pdpyqkrb-benchmark-$suffix data/v15/pdpyqkrb-$suffix; \
    done

COPY . .

RUN . /venv/main/bin/activate \
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

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV VASTAI=1

ENTRYPOINT ["/root/entrypoint.sh"]
