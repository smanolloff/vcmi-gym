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
    env=None,
    #env=dict(
    #    # opponent="BattleAI",
    #    opponent="StupidAI",
    #    mapname="gym/generated/4096/4x1024.vmap",
    #    max_steps=1000,
    #    random_heroes=1,
    #    random_obstacles=1,
    #    town_chance=30,
    #    warmachine_chance=40,
    #    random_terrain_chance=100,
    #    tight_formation_chance=20,
    #    allow_invalid_actions=True,
    #    user_timeout=3600,
    #    vcmi_timeout=3600,
    #    boot_timeout=300,
    #    conntype="thread",
    #    # vcmi_loglevel_global="trace",
    #    # vcmi_loglevel_ai="trace",
    #),

    # s3data=None,
    s3data=dict(
        bucket_name="vcmi-gym",
        s3_prefix="v8",
        # Don't store in config (will appear in clear text in config.json)
        # aws_access_key=os.environ["AWS_ACCESS_KEY"],
        # aws_secret_key=os.environ["AWS_SECRET_KEY"],
        region_name="eu-north-1",
        cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache")),
        num_workers=1,
        prefetch_factor=1,
        shuffle=True,
    ),

    train={
        # TODO: consider torch.optim.lr_scheduler.StepLR
        "learning_rate": 1e-4,

        "buffer_capacity": 10_000,
        "train_epochs": 1,
        "train_batch_size": 1000,
        "eval_env_steps": 10_000,

        # !!! DEBUG (linter warning is OK) !!!
        "buffer_capacity": 10000,
        "train_epochs": 100,
        "train_batch_size": 2000,
        "eval_env_steps": 100,
    }
)

