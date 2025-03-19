import os

#
# Depending on the `env` and `s3` configs, the behaviour is:
#
# env=no    s3=no   => ERROR
# env=yes   s3=no   => sample from env
# env=no    s3=yes  => load samples from S3
# env=yes   s3=yes  => sample from env + save to S3
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

    # s3=None,
    s3=dict(
        checkpoint=dict(
            bucket_name="vcmi-gym",
            s3_dir="models",
        ),
        data=dict(
            bucket_name="vcmi-gym",
            s3_dir="v8-100k",
            cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache")),
            cached_files_max=None,
            num_workers=1,
            prefetch_factor=1,
            shuffle=True,
        ),
    ),

    train={
        "eval_every": 10,

        # TODO: consider torch.optim.lr_scheduler.StepLR
        "learning_rate": 1e-4,

        "buffer_capacity": 10_000,
        "train_epochs": 1,
        "train_batch_size": 2000,
        "eval_env_steps": 10_000,

        # !!! DEBUG (linter warning is OK) !!!
        "buffer_capacity": 10000,
        "train_epochs": 10,
        "train_batch_size": 2000,
        "eval_env_steps": 100,
    }
)

