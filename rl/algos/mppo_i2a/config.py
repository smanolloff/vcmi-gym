import os

env_kwargs = dict(
    random_stack_chance=0,
    role="defender",
    random_terrain_chance=100,
    tight_formation_chance=0,
    max_steps=500,
    vcmi_loglevel_global="error",
    vcmi_loglevel_ai="error",
    vcmienv_loglevel="WARN",
    random_heroes=1,
    random_obstacles=1,
    town_chance=10,
    warmachine_chance=40,
    mana_min=0,
    mana_max=0,
    reward_step_fixed=-1,
    reward_dmg_mult=1,
    reward_term_mult=1,
    swap_sides=0,
    user_timeout=1800,
    vcmi_timeout=60,
    boot_timeout=30,
)

config = dict(
    name_template="{datetime}-{id}-v12",
    out_dir_template="data/world/mppo-i2a",

    # XXX: s3_dir's "{wandb_group}" substring will be replaced with this value
    wandb_group="mppo-i2a",
    wandb_log_interval_s=60,

    checkpoint=dict(
        interval_s=0,  # use 0 to checkpoint on each eval
        permanent_interval_s=6*3600,  # 6h (use int(2e9) to disable)
        optimize_local_storage=False,
        s3=dict(
            bucket_name="vcmi-gym",
            s3_dir="{wandb_group}/models"
        ),
    ),
    eval=dict(
        env=dict(
            num_envs=40,
            kwargs=dict(env_kwargs, mapname="gym/generated/evaluation/8x512.vmap")
        ),
        num_vsteps=100,
        interval_s=1800,
    ),
    train=dict(
        env=dict(
            # XXX: more venvs = more efficient GPU usage (B=num_envs)
            # XXX: 50 envs ~= 30G RAM
            num_envs=40,
            kwargs=dict(env_kwargs, mapname="gym/generated/4096/4x1024.vmap")
        ),
        # XXX: ep_len_mean=30... try to capture AT LEAST 1 episode per env
        num_vsteps=40,  # num_steps = num_vsteps * num_envs
        num_minibatches=50,
        update_epochs=1,

        gamma=0.85,
        gae_lambda=0.9,
        ent_coef=0.05,
        clip_coef=0.5,
        learning_rate=1e-4,
        vf_coef=0.5,
        norm_adv=True,
        clip_vloss=True,
        target_kl=None,
        max_grad_norm=1,
        weight_decay=0.05,
        distill_lambda=1.0,

        torch_autocast=True,
    ),
    model=dict(
        i2a_fc_units=1024,
        num_trajectories=20,  # valid actions are ~65 on average... (25 for peasant)
        rollout_dim=1024,
        rollout_policy_fc_units=1024,
        horizon=3,
        obs_processor_output_size=2048,
        transition_model_file="hauzybxn-model.pt",
        action_prediction_model_file="ogyesvkb-model.pt",
        reward_prediction_model_file="aexhrgez-model.pt",

        # Num transitions after 10K steps on:
        #   (4x1024, defender, vs. BattleAI, swap=0, heroes=1, obst=100, mach=40, town=10, stack=0):
        #
        # 1: 1.8% (180)             // no action (~50% of resets, where we are first)
        # 2: 39.0% (3899)           // no enemy action
        # 3: 31.9% (3192)           // 1 enemy action
        # 4: 10.7% (1067)           // ...
        # --------------- = 83%
        # 5: 6.4% (637)
        # 6: 3.7% (365)
        # 7: 2.5% (249)
        # 8: 1.6% (156)
        # 9: 0.9% (87)
        # 10+: < 1%
        max_transitions=5,
    ),
)

config["checkpoint"]["s3"]["s3_dir"] = config["checkpoint"]["s3"]["s3_dir"].replace("{wandb_group}", config["wandb_group"])

# Debug
if os.getenv("VASTAI", None) != "1":
    config["train"]["num_vsteps"] = 4
    config["train"]["num_minibatches"] = 2
    config["train"]["update_epochs"] = 1
    config["train"]["env"]["num_envs"] = 1
    config["train"]["env"]["kwargs"]["mapname"] = "gym/A1.vmap"
    config["eval"]["num_vsteps"] = 2
    config["eval"]["env"]["num_envs"] = 1
    config["eval"]["env"]["kwargs"]["mapname"] = "gym/A1.vmap"
    config["eval"]["interval_s"] = 10
    config["wandb_log_interval_s"] = 5
    config["checkpoint"]["interval_s"] = 11
    config["model"].update(
        i2a_fc_units=16,
        num_trajectories=1,  # valid actions are ~65 on average... (25 for peasant)
        rollout_dim=16,
        rollout_policy_fc_units=16,
        horizon=2,
        obs_processor_output_size=16,
    )
