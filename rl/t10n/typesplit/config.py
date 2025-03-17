import os

#
# Depending on the `env` and `s3data` configs, the behaviour is:
#
# env=no    s3data=no   => ERROR
# env=yes   s3data=no   => sample from env
# env=no    s3data=yes  => load samples from S3
# env=yes   s3data=yes  => sample from env + save to S3
#
# NOTE: saving to S3 in this script is not implemented
#       Files are saved to local disk only (see `bufdir`)
#       and can be later uploaded via s3uploader.py
#

config = dict(
    # env=None,
    env=dict(
        # opponent="BattleAI",  # BROKEN in develop1.6 from 2025-01-31
        opponent="StupidAI",
        mapname="gym/generated/4096/4x1024.vmap",
        # mapname="gym/A1.vmap",
        max_steps=1000,
        random_heroes=1,
        random_obstacles=1,
        town_chance=30,
        warmachine_chance=40,
        random_terrain_chance=100,
        tight_formation_chance=20,
        allow_invalid_actions=True,
        user_timeout=3600,
        vcmi_timeout=3600,
        boot_timeout=300,
        conntype="thread",
        # vcmi_loglevel_global="trace",
        # vcmi_loglevel_ai="trace",
    ),

    # s3data=None,
    s3data=dict(
        bucket_name="vcmi-gym",
        s3_prefix="v8",
        # Must not be part of the config (clear-text in config.json)
        # aws_access_key=os.environ["AWS_ACCESS_KEY"],
        # aws_secret_key=os.environ["AWS_SECRET_KEY"],
        region_name="eu-north-1",
        cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache")),
        num_workers=1,
        prefetch_factor=1,
        shuffle=True,
    ),

    train={
        "lr_start": 1e-2,
        "lr_min": 1e-3,
        "lr_step_size": 60,  # 1step ~= 1min if epochs=3 and buf=10K
        "lr_gamma": 0.75,

        "buffer_capacity": 10_000,
        "train_epochs": 1,
        "train_batch_size": 1000,
        "eval_env_steps": 10_000,

        # !!! DEBUG (linter warning is OK) !!!
        "buffer_capacity": 100,
        "train_epochs": 1,
        "train_batch_size": 10,
        "eval_env_steps": 100,
    }
)
