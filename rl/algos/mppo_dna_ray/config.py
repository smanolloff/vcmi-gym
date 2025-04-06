# XXX:
# A/ To RESUME a run, set `agent_load_file` here
#
# B/ To LOAD a file into a NEW run, pass "--resume <json_file>"
#   * `agent_load_file` will be overwritten
#   * change config values by editing the JSON file
#     NOTE: specify the changed arg NAMES in `overwrite`, e.g. ["env.max_steps"]

config = dict(
    # run_id        # always auto
    # resume        # must use the default (False) for new runs
    # overwrite     # must use the default ([]) for new runs

    run_name_template="{datetime}-{id}",
    group_id="mppo_dna_ray",

    agent_load_file=None,
    # agent_load_file="data/mppo_dna_ray/hktcyplj-agent-1743934056.pt",

    save_every=10,
    max_old_saves=0,

    loglevel="DEBUG",
    run_name=None,
    trial_id=None,
    wandb_project="vcmi-gym",
    notes=None,
    vsteps_total=0,
    seconds_total=0,
    rollouts_per_log=1,
    success_rate_target=None,
    ep_rew_mean_target=None,
    quit_on_target=False,
    mapside="attacker",
    permasave_every=int(2e9),  # disable with int(2e9), which is always > time.time()
    out_dir_template="data/{group_id}",
    opponent_load_file=None,
    opponent_sbm_probs=[1, 0, 0],
    weight_decay=0.05,
    lr_schedule=dict(mode="const", start=0.0001),
    lr_schedule_value=dict(mode="const", start=0.0001),
    lr_schedule_policy=dict(mode="const", start=0.0001),
    lr_schedule_distill=dict(mode="const", start=0.0001),
    num_steps_per_sampler=200,  # num_steps = num_steps_per_sampler * num_samplers
    num_samplers=1,
    gamma=0.85,
    gae_lambda=0.9,
    gae_lambda_policy=0.95,
    gae_lambda_value=0.95,
    num_minibatches=2,
    num_minibatches_value=2,
    num_minibatches_policy=2,
    num_minibatches_distill=2,
    update_epochs=2,
    update_epochs_value=2,
    update_epochs_policy=2,
    update_epochs_distill=2,
    norm_adv=True,
    clip_coef=0.5,
    clip_vloss=True,
    ent_coef=0.05,
    max_grad_norm=1,
    distill_beta=1.0,
    target_kl=None,
    logparams={},
    cfg_file=None,
    seed=42,
    skip_wandb_init=False,
    skip_wandb_log_code=False,
    envmaps=["gym/generated/4096/4x1024.vmap"],
    env=dict(
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
        user_timeout=0,
        vcmi_timeout=0,
        boot_timeout=0,
        conntype="thread"
    ),
    # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
    env_wrappers=[dict(module="vcmi_gym.envs.util.wrappers", cls="LegacyObservationSpaceWrapper")],
    env_version=10,
    network={
        "encoders": {
            "other": {"blocks": 3, "size": 10},
            "hex": {"blocks": 3, "size": 10},
            "merged": {"blocks": 3, "size": 10},
        },
        "heads": {
            "actor": {"size": 2312},
            "critic": {"size": 1}
        }
    }
)
