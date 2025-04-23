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
env_kwargs = dict(
    opponent="StupidAI",
    max_steps=1000,
    random_heroes=1,
    random_obstacles=1,
    swap_sides=1,
    town_chance=30,
    warmachine_chance=40,
    random_terrain_chance=100,
    tight_formation_chance=20,
    allow_invalid_actions=True,
    user_timeout=3600,
    vcmi_timeout=3600,
    boot_timeout=300,
    vcmi_loglevel_global="error",
    vcmi_loglevel_ai="error",
)

config = dict(
    name_template="{datetime}-{id}-v12-swap-T-E512_H8_L6-B300-RTX3080",
    out_dir_template="data/world/t10n",
    wandb_group="transition-model",

    env=dict(
        train=dict(
            # num_workers=10,
            # batch_size=1000,
            num_workers=1,
            batch_size=100,  # buffer capacity = num_workers * batch_size
            prefetch_factor=1,
            kwargs=dict(env_kwargs, mapname="gym/A1.vmap")
            # kwargs=dict(env_kwargs, mapname="gym/generated/4096/4x1024.vmap")
        ),
        eval=dict(
            # num_workers=1,
            # batch_size=1000,  # buffer capacity = num_workers * batch_size
            num_workers=1,
            batch_size=200,  # buffer capacity = num_workers * batch_size
            prefetch_factor=1,
            kwargs=dict(env_kwargs, mapname="gym/A1.vmap"),
            # kwargs=dict(env_kwargs, mapname="gym/generated/evaluation/8x512.vmap"),
        ),
    ),

    checkpoint_interval_s=900,  # NOTE: checked only after eval
    permanent_checkpoint_interval_s=6*3600,  # 6h (use int(2e9) to disable)

    s3=dict(
        optimize_local_storage=False,

        # checkpoint=None,
        checkpoint=dict(
            bucket_name="vcmi-gym",
            s3_dir="models",
        ),

        # data=dict(
        #     train=dict(
        #         bucket_name="vcmi-gym",
        #         s3_dir="v12/4x1024",
        #         cache_dir=os.path.abspath("data/.s3_cache"),
        #         cached_files_max=None,
        #         num_workers=1,
        #         batch_size=1000,  # buffer capacity = num_workers * batch_size
        #         prefetch_factor=1,
        #         pin_memory=False,       # causes hangs when enabled
        #         shuffle=False,
        #     ),
        #     eval=dict(
        #         bucket_name="vcmi-gym",
        #         s3_dir="v12/8x512",
        #         cache_dir=os.path.abspath("data/.s3_cache"),
        #         cached_files_max=None,
        #         num_workers=1,
        #         batch_size=1000,  # buffer capacity = num_workers * batch_size
        #         prefetch_factor=1,
        #         pin_memory=False,       # causes hangs when enabled
        #         shuffle=False,
        #     )
        # ),
    ),

    eval=dict(
        interval_s=60,  # wandb_log will also be called here
        # batch_size=200,
        batch_size=20,
    ),
    train=dict(
        accumulate_grad=False,  # makes 1 batch = entire buffer
        # batch_size=250,
        batch_size=25,
        learning_rate=1e-4,
        epochs=1,
    )
)
