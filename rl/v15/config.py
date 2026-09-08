import os

eval_interval_s = 3600

train_env_kwargs = dict(
    mapname="gym/ml-mini.vmap",
    role="defender",
    # opponent="",  # overwritten
    max_steps=500,
    vcmi_loglevel_global="error",
    vcmi_loglevel_ai="error",
    vcmienv_loglevel="WARN",
    random_heroes=0,
    random_obstacles=1,
    random_terrain_chance=100,
    uniform_chance=0,
    opponent_uniform_chance=0,
    whitelist="",
    opponent_whitelist="",
    tight_formation_chance=0,
    creature_bank_chance=0,
    town_chance=10,
    mirror_armies=False,
    random_armies=True,
    random_army_value_min=500,
    random_army_value_max=500_000,
    random_army_target_var=30,
    warmachine_chance=40,
    mana_min=0,
    mana_max=0,
    random_primary_skills=0,
    reward_step_fixed=-0.5,

    # These require BATTLE_ROUND in obs
    reward_prog_base=0.1,
    reward_prog_trigger=10,
    reward_prog_exponent=2,
    reward_prog_limit=16,

    reward_dmg_mult=0.01,
    reward_term_mult=0.01,
    reward_relval_mult=0.01,
    swap_sides=0,

    # ignored_edges=[],
    # If keys are lists, they will be converted to tuples by VcmiEnv
    # Prefer lists here (for consistence with json-serialized configs)
    ignored_edges=[
        # These often dominate the graph (e.g. 30k of total 40k edges)
        ["Hex", "BecomesMeleeTargetAfter", "Action"],
        ["Hex", "BecomesShootTargetAfter", "Action"],
    ],

    # With DualVecEnv, all timeouts must be the same (large enough)
    user_timeout=2400,
    vcmi_timeout=2400,
    boot_timeout=2400,
)

eval_env_kwargs = dict(
    train_env_kwargs,
    mapname="gym/ml-eval.vmap",
    # Turn everything off by default => env variants must explicitly enable those
    random_heroes=0,
    town_chance=0,
    mirror_armies=False,
    random_armies=False,
    user_timeout=300 + eval_interval_s,  # must be >= eval.interval_s + 300
    vcmi_timeout=300 + eval_interval_s,  # must be >= eval.interval_s + 300

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

config = dict(
    version=15,
    name_template="{datetime}-{id}-{suffix}",
    out_dir_template="data/v15",

    # XXX: s3_dir's "{wandb_group}" substring will be replaced with this value
    wandb_group="v15",
    wandb_log_interval_s=60,

    checkpoint=dict(
        # Non-permanent checkpoint is made after eval if result is good
        # Permanent checkpoint is made every X seconds, regardless of eval result
        permanent_interval_s=12*3600,  # disable with int(2e9)
        volatile_interval_s=3600,  # disable with int(2e9)
        volatile_num_tags=2,  # disable with int(2e9)
        optimize_local_storage=False,
        s3=dict(
            bucket_name="vcmi-gym",
            s3_dir="{wandb_group}/models"
        ),
    ),
    eval=dict(
        interval_s=eval_interval_s,
        env_variants={
            "BattleAI.open": dict(
                # XXX: too verbose
                num_vsteps=2500,
                env_meta=dict(
                    type="BattleAI",
                    num=10,
                    kwargs=dict(eval_env_kwargs, random_heroes=1)
                )
            ),
            "BattleAI.vip": dict(
                num_vsteps=2500,
                env_meta=dict(
                    type="VIPBot",
                    num=10,
                    kwargs=dict(eval_env_kwargs, random_armies=True)
                )
            ),
            "BattleAI.har": dict(
                num_vsteps=2500,
                env_meta=dict(
                    type="HARBot",
                    num=10,
                    kwargs=dict(eval_env_kwargs, random_armies=True)
                )
            ),
            "BattleAI.bank": dict(
                num_vsteps=2500,
                env_meta=dict(
                    type="HARBot",
                    num=1,
                    kwargs=dict(
                        eval_env_kwargs,
                        random_armies=True,
                        creature_bank_chance=100,
                        uniform_chance=100,
                        whitelist="core:griffin",
                        opponent_whitelist="core:angel,core:pikeman",
                        # NOTE: army values use CalculateValue() in ServePlugin.cpp
                        random_army_value_min=7500, # 50 griffins
                        random_army_value_max=30_000,  # 200 griffins
                        random_army_target_var=20,  # 20% var on 7500 =~ +-3000 (1 angel=2500)
                        # These are always disabled on banks, set them for consistency
                        town_chance=0,
                        warmachine_chance=0,
                        random_obstacles=0,
                    )
                )
            ),
            "MMAI.open": dict(
                num_vsteps=500,
                env_meta=dict(
                    type="torch_model",
                    num=10,
                    kwargs=dict(eval_env_kwargs, random_heroes=1),
                    model_=dynamic_bot("pdpyqkrb", 7200)
                )
            )
        }
    ),
    train=dict(
        env_metas=[
            # XXX: the total sum of all train envs must be divisible by num_vsteps * num_minibatches
            dict(type="BattleAI", num=4, kwargs=dict(train_env_kwargs)),
            dict(type="VIPBot", num=5, kwargs=dict(train_env_kwargs)),
            dict(type="HARBot", num=5, kwargs=dict(train_env_kwargs)),
            dict(type="HARBot", num=1, kwargs=dict(
                train_env_kwargs,
                creature_bank_chance=100,
                uniform_chance=100,
                whitelist="core:griffin",
                opponent_whitelist="core:angel,core:pikeman",
                random_army_value_min=7500,  # 50 griffins
                random_army_value_max=30_000,  # 200 griffins
                random_army_target_var=20,  # 20% var on 7500 =~ +-3000 (1 angel=2500)
                # These are always disabled on banks, set them for consistency
                town_chance=0,
                warmachine_chance=0,
                random_obstacles=0,
            )),
            dict(type="torch_model", num=5, kwargs=dict(train_env_kwargs), model_=dynamic_bot("pdpyqkrb", 7200)),
        ],

        num_vsteps=150,                 # num_steps = num_vsteps * num_envs
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
    ),
    model=dict(
        policy_head_hidden_channels=128,
        value_head_hidden_channels=128,
        gnn_num_layers=3,
        gnn_hidden_channels=128,
        gnn_out_channels=64,

        gnn_conv_cls="GENConv",
        gnn_conv_kwargs=dict(
            aggr="softmax",
            learn_t=True,
            num_layers=2,  # number of MLP layers (within a single GNN layer)
            norm=None,  # already applying LayerNorm externally (with the residual)
        ),

        # gnn_conv_cls="GATConv",
        # gnn_conv_kwargs=dict(
        #     heads=4,
        #     concat=False,
        #     dropout=0.0,
        #     add_self_loops=False,
        # ),

        # gnn_conv_cls="ResGatedGraphConv",
        # gnn_conv_kwargs=dict(
        #     root_weight=False,
        # ),
    ),


)

config["checkpoint"]["s3"]["s3_dir"] = config["checkpoint"]["s3"]["s3_dir"].replace("{wandb_group}", config["wandb_group"])

# Debug
if os.getenv("VASTAI", None) != "1":
    config["train"]["num_vsteps"] = 40
    config["train"]["num_minibatches"] = 4
    config["train"]["update_epochs"] = 2

    # torch_model envs force a download from s3
    config["eval"]["env_variants"] = {k: v for k, v in config["eval"]["env_variants"].items() if v["env_meta"]["type"] != "torch_model"}
    config["train"]["env_metas"] = [em for em in config["train"]["env_metas"] if em["type"] != "torch_model"]

    for env_meta in config["train"]["env_metas"]:
        env_meta["num"] = min(env_meta["num"], 2)
        env_meta["kwargs"]["mapname"] = "gym/A1.vmap"
        # env_meta["kwargs"]["vcmienv_loglevel"] = "DEBUG"

    # env_variants={
    #     "BattleAI.open": dict(
    #         # XXX: too verbose
    #         num_vsteps=2500,
    #         env_meta=dict(
    #             type="BattleAI",
    #             num=10,
    #             kwargs=dict(eval_env_kwargs, random_heroes=1)
    #         )
    #     ),

    for name, varcfg in config["eval"]["env_variants"].items():
        varcfg["num_vsteps"] = 40
        varcfg["env_meta"]["num"] = min(varcfg["env_meta"]["num"], 2)
        varcfg["env_meta"]["kwargs"]["warmachine_chance"] = 0
        varcfg["env_meta"]["kwargs"]["mapname"] = "gym/A1.vmap"
        # varcfg["env_meta"]["kwargs"]["vcmienv_loglevel"] = "DEBUG"

    config["eval"]["interval_s"] = 30
    config["wandb_log_interval_s"] = 30

    config["model"]["gnn_num_layers"] = 3
    config["model"]["gnn_hidden_channels"] = 64
    config["model"]["gnn_out_channels"] = 32
    config["model"]["value_head_hidden_channels"] = 64
    config["model"]["policy_head_hidden_channels"] = 64

    config["checkpoint"]["permanent_interval_s"] = 300
    config["checkpoint"]["volatile_interval_s"] = 100
