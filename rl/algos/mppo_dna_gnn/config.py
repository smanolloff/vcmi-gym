import os

train_env_kwargs = dict(
    role="defender",
    # opponent="",  # overwritten
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
    reward_step_fixed=-0.5,
    reward_dmg_mult=0.01,
    reward_term_mult=0.01,
    reward_relval_mult=0.01,
    swap_sides=0,
    # With DualVecEnv, all timeouts must be the same (large enough)
    user_timeout=7200,
    vcmi_timeout=7200,
    boot_timeout=7200,
)

# Example:
# static_bot("data/mppo-dna-heads/nkjrmrsq-202509291846")
static_bot = lambda path_prefix: dict(
    type="static",
    config_file=f"{path_prefix}-config.json",
    weights_file=f"{path_prefix}-model-dna.pt",
)

# Example:
# dynamic_bot("nkjrmrsq", 7200)
dynamic_bot = lambda run_id, reload_interval_s: dict(
    type="dynamic",
    run_id=run_id,
    reload_interval_s=reload_interval_s
)

gen_num_envs = lambda StupidAI, BattleAI, model: dict(StupidAI=StupidAI, BattleAI=BattleAI, model=model)

eval_variant = lambda num_envs_per_opponent, model, **env_kwargs: dict(
    num_envs_per_opponent=num_envs_per_opponent,
    kwargs=dict(
        train_env_kwargs,
        mapname="gym/generated/evaluation/8x512.vmap",
        random_stack_chance=0,
        **env_kwargs,
    ),
    model=model,
)

config = dict(
    name_template="{datetime}-{id}-v12",
    out_dir_template="data/mppo-dna-heads",

    # XXX: s3_dir's "{wandb_group}" substring will be replaced with this value
    wandb_group="mppo-dna-heads",
    wandb_log_interval_s=60,

    checkpoint=dict(
        # Non-permanent checkpoint is made after eval if result is good
        # Permanent checkpoint is made every X seconds, regardless of eval result
        permanent_interval_s=12*3600,  # disable with int(2e9)
        optimize_local_storage=False,
        s3=dict(
            bucket_name="vcmi-gym",
            s3_dir="{wandb_group}/models"
        ),
    ),
    eval=dict(
        env_variants={
            # "BattleAI.town": eval_variant(gen_num_envs(0, 2, 0), town_chance=100),
            # "BattleAI.open": eval_variant(
            #     num_envs_per_opponent=gen_num_envs(0, 2, 0),
            #     model=None,
            #     town_chance=0,
            # ),
            "MMAI.open": eval_variant(
                num_envs_per_opponent=gen_num_envs(0, 0, 1),
                model=dynamic_bot("nkjrmrsq", 5),
                # model=static_bot("nkjrmrsq-202509291846"),
                town_chance=0,
            ),
        },
        num_vsteps=10_000,
        interval_s=1800,
        at_script_start=True,
    ),
    train=dict(
        env=dict(
            # XXX: more venvs = more efficient GPU usage (B=num_envs)
            # XXX: 50 envs ~= 30G RAM
            kwargs=dict(train_env_kwargs, mapname="gym/generated/4096/4x1024.vmap"),
            # num_envs_per_opponent=dict(StupidAI=0, BattleAI=40, model=0),

            num_envs_per_opponent=dict(StupidAI=0, BattleAI=0, model=1),
            # model=static_bot("data/mppo-dna-heads/tukbajrv-202509241418"),
            model=static_bot("nkjrmrsq-202509291846"),
        ),
        num_vsteps=125,                 # num_steps = num_vsteps * num_envs
        num_minibatches=20,             # mb_size = num_steps / num_minibatches
        update_epochs=2,

        learning_rate=1e-4,
        lr_scheduler_mod="torch.optim.lr_scheduler",
        lr_scheduler_cls="LinearLR",
        lr_scheduler_kwargs=dict(start_factor=1, end_factor=1e-1, total_iters=100),
        lr_scheduler_interval_s=600,

        gamma=0.95,
        gae_lambda=0.8,
        ent_coef=0.03,
        clip_coef=0.6,
        norm_adv=True,
        clip_vloss=True,
        target_kl=None,
        max_grad_norm=1,
        distill_beta=1.0,

        torch_autocast=False,
        torch_cuda_matmul=False,
        torch_detect_anomaly=False,
        torch_compile=False,
    ),
    model=dict(
        gnn_num_layers=3,
        gnn_out_channels=128,
        gnn_hidden_channels=256,
        critic_hidden_features=256,
        # result_predictor_hidden_features=256,
    ),
)

config["checkpoint"]["s3"]["s3_dir"] = config["checkpoint"]["s3"]["s3_dir"].replace("{wandb_group}", config["wandb_group"])

# Debug
if os.getenv("VASTAI", None) != "1":
    config["train"]["num_vsteps"] = 40
    config["train"]["num_minibatches"] = 4
    config["train"]["update_epochs"] = 2
    config["train"]["env"]["num_envs_per_opponent"] = {k: min(v, 2) for k, v in config["train"]["env"]["num_envs_per_opponent"].items()}
    config["train"]["env"]["kwargs"]["mapname"] = "gym/A1.vmap"
    # config["train"]["env"]["kwargs"]["vcmienv_loglevel"] = "DEBUG"

    config["eval"]["num_vsteps"] = 100
    # config["eval"]["env_variants"] = dict(list(config["eval"]["env_variants"].items())[:1])
    for name, envcfg in config["eval"]["env_variants"].items():
        envcfg["num_envs_per_opponent"] = {k: min(v, 1) for k, v in envcfg["num_envs_per_opponent"].items()}
        envcfg["kwargs"]["warmachine_chance"] = 0
        envcfg["kwargs"]["mapname"] = "gym/A1.vmap"
        # envcfg["kwargs"]["vcmienv_loglevel"] = "DEBUG"

    # DEBUG dual vec env:
    config["train"]["env"]["kwargs"] = list(config["eval"]["env_variants"].values())[0]["kwargs"]

    config["eval"]["interval_s"] = 300
    config["wandb_log_interval_s"] = 180

    config["model"]["gnn_num_layers"] = 3
    config["model"]["gnn_num_heads"] = 1
    config["model"]["gnn_hidden_channels"] = 32
    config["model"]["gnn_out_channels"] = 16
    config["model"]["critic_hidden_features"] = 16
    config["model"]["result_predictor_hidden_features"] = 16
    config["model"]["z_size_other"] = 16
    config["model"]["z_size_merged"] = 50
