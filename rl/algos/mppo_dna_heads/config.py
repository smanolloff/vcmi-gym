import os

train_env_kwargs = dict(
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
    random_stack_chance=20,  # makes armies unbalanced
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

eval_variant = lambda **env_kwargs: dict(
    num_envs=5,
    sync=False,
    kwargs=dict(
        train_env_kwargs,
        mapname="gym/generated/evaluation/8x512.vmap",
        random_stack_chance=0,
        **env_kwargs,
    )
)

config = dict(
    name_template="{datetime}-{id}-v12",
    out_dir_template="data/mppo-dna-heads",

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
        env_variants={
            "StupidAI.town": eval_variant(opponent="StupidAI", town_chance=100),
            "StupidAI.open": eval_variant(opponent="StupidAI", town_chance=0),
            "BattleAI.town": eval_variant(opponent="BattleAI", town_chance=100),
            "BattleAI.open": eval_variant(opponent="BattleAI", town_chance=0),
        },
        num_vsteps=1000,
        interval_s=1800,
        at_script_start=False,
    ),
    train=dict(
        env=dict(
            # XXX: more venvs = more efficient GPU usage (B=num_envs)
            # XXX: 50 envs ~= 30G RAM
            num_envs=100,
            sync=False,
            kwargs=dict(train_env_kwargs, mapname="gym/generated/4096/4x1024.vmap"),
        ),
        num_vsteps=100,                 # num_steps = num_vsteps * num_envs
        num_minibatches=20,             # mb_size = num_steps / num_minibatches
        update_epochs=2,

        learning_rate=1e-4,
        lr_scheduler_mod="torch.optim.lr_scheduler",
        lr_scheduler_cls="LinearLR",
        lr_scheduler_kwargs=dict(start_factor=1, end_factor=1e-2, total_iters=100),
        lr_scheduler_interval_s=600,

        gamma=0.95,
        gae_lambda=0.8,
        ent_coef=0.03,
        clip_coef=0.6,
        norm_adv=True,
        clip_vloss=False,
        target_kl=None,
        max_grad_norm=4,
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
    config["train"]["num_vsteps"] = 16
    config["train"]["num_minibatches"] = 4
    config["train"]["update_epochs"] = 2
    config["train"]["env"]["num_envs"] = 1
    config["train"]["env"]["kwargs"]["mapname"] = "gym/A1.vmap"
    config["eval"]["num_vsteps"] = 100

    config["eval"]["env_variants"] = dict(list(config["eval"]["env_variants"].items())[:1])
    for name, envcfg in config["eval"]["env_variants"].items():
        envcfg["num_envs"] = 1
        envcfg["kwargs"]["mapname"] = "gym/A1.vmap"

    config["eval"]["interval_s"] = 30
    config["wandb_log_interval_s"] = 5
    config["checkpoint"]["interval_s"] = 10

    config["model"]["z_size_other"] = 4
    config["model"]["z_size_hex"] = 4
    config["model"]["z_size_merged"] = 16
