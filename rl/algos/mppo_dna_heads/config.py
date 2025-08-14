import os

env_kwargs = dict(
    role="defender",
    opponent="StupidAI",
    max_steps=500,
    vcmi_loglevel_global="error",
    vcmi_loglevel_ai="error",
    vcmienv_loglevel="WARN",
    random_heroes=1,
    random_obstacles=1,
    random_terrain_chance=100,
    tight_formation_chance=0,
    town_chance=10,
    random_stack_chance=20,
    warmachine_chance=40,
    mana_min=0,
    mana_max=0,
    # reward_step_fixed=-1,
    # reward_dmg_mult=1,
    # reward_term_mult=1,
    # reward_relval_mult=1,
    reward_step_fixed=-0.01,
    reward_dmg_mult=0.01,
    reward_term_mult=0.01,
    reward_relval_mult=0.01,
    swap_sides=0,
    user_timeout=7200,
    vcmi_timeout=60,
    boot_timeout=30,
)

config = dict(
    name_template="{datetime}-{id}-v12",
    out_dir_template="data/world/mppo-dna-heads",

    # XXX: s3_dir's "{wandb_group}" substring will be replaced with this value
    wandb_group="mppo-dna-heads",
    wandb_log_interval_s=60,

    checkpoint=dict(
        interval_s=3600,
        permanent_interval_s=12*3600,  # disable with int(2e9)
        optimize_local_storage=False,
        s3=dict(
            bucket_name="vcmi-gym",
            s3_dir="{wandb_group}/models"
        ),
    ),
    eval=dict(
        env=dict(
            num_envs=1,
            kwargs=dict(env_kwargs, mapname="gym/generated/evaluation/8x512.vmap")
        ),
        num_vsteps=1000,
        interval_s=1800,
        at_script_start=False,
    ),
    train=dict(
        env=dict(
            # XXX: more venvs = more efficient GPU usage (B=num_envs)
            # XXX: 50 envs ~= 30G RAM
            num_envs=10,
            kwargs=dict(env_kwargs, mapname="gym/generated/4096/4x1024.vmap")
        ),
        # XXX: ep_len_mean=30... try to capture AT LEAST 1 episode per env
        num_vsteps=100,  # num_steps = num_vsteps * num_envs
        num_minibatches=20,
        update_epochs=1,

        learning_rate=1e-4,
        # lr_scheduler_interval_s=5,  # use 1e9 to disable
        lr_scheduler_interval_s=1e9,  # use 1e9 to disable
        lr_scheduler_step_mult=0.8,    # => 1e-4...1e-5 in 10 steps
        lr_scheduler_min_value=1e-4,

        gamma=0.8,
        gae_lambda=0.9,
        ent_coef=0.01,
        clip_coef=0.3,
        vf_coef=0.5,
        norm_adv=True,
        clip_vloss=True,
        target_kl=None,
        max_grad_norm=0.5,
        distill_beta=1.0,

        torch_autocast=False,
    ),
    model=dict(
        z_size_other=32,
        z_size_hex=32,
        z_size_merged=1024
    ),
)

config["checkpoint"]["s3"]["s3_dir"] = config["checkpoint"]["s3"]["s3_dir"].replace("{wandb_group}", config["wandb_group"])

# Debug
if os.getenv("VASTAI", None) != "1":
    config["train"]["num_vsteps"] = 256
    config["train"]["num_minibatches"] = 4
    config["train"]["update_epochs"] = 2
    config["train"]["env"]["num_envs"] = 1
    config["train"]["env"]["kwargs"]["mapname"] = "gym/A1.vmap"
    config["eval"]["num_vsteps"] = 500
    config["eval"]["env"]["num_envs"] = 1
    config["eval"]["env"]["kwargs"]["mapname"] = "gym/A1.vmap"

    config["eval"]["interval_s"] = 30
    config["wandb_log_interval_s"] = 5
    config["checkpoint"]["interval_s"] = 10
