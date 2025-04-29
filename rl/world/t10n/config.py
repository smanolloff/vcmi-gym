import os

from .weights import weights

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
            num_workers=1,
            batch_size=4,  # buffer capacity = num_workers * batch_size
            prefetch_factor=1,
            kwargs=dict(env_kwargs, mapname="gym/A1.vmap")
        ),
        eval=dict(
            num_workers=1,
            batch_size=4,  # buffer capacity = num_workers * batch_size
            prefetch_factor=1,
            kwargs=dict(env_kwargs, mapname="gym/A1.vmap"),
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

    train=dict(
        accumulate_grad=False,  # makes 1 batch = entire buffer
        batch_size=2,
        learning_rate=1e-4,
        epochs=1,
    ),
    eval=dict(
        interval_s=5,
        batch_size=2,
    ),
    wandb_log_interval_s=5,
    wandb_table_update_interval_s=8,  # data will be added here (rows=2*NUM_ATTRS)
    wandb_table_log_interval_s=20,    # table will be uploaded here (creates new W&B artifact version)
    wandb_table_log=True,
    weights=weights
)

if os.getenv("VASTAI", None) == "1":
    if config.get("env", {}).get("train"):
        config["env"]["train"]["num_workers"] = 6
        config["env"]["train"]["batch_size"] = 1000
        config["env"]["train"]["kwargs"]["mapname"] = "gym/generated/4096/4x1024.vmap"

    if config.get("env", {}).get("eval"):
        config["env"]["eval"]["num_workers"] = 1
        config["env"]["eval"]["batch_size"] = 5000
        config["env"]["eval"]["kwargs"]["mapname"] = "gym/generated/evaluation/8x512.vmap"

    config["train"]["batch_size"] = 250
    config["eval"]["batch_size"] = 200
    config["eval"]["interval_s"] = 60

    config["wandb_log_interval_s"] = 60
    config["wandb_table_update_interval_s"] = 600
    config["wandb_table_log_interval_s"] = 3600
