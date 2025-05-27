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
    group_id="mppo_dna_i2a",

    agent_load_file=None,
    # agent_load_file="data/mppo_dna_ray/hktcyplj-agent-1743934056.pt",

    save_every=600,
    max_old_saves=0,

    num_samplers=2,
    num_steps_per_sampler=200,  # num_steps = num_steps_per_sampler * num_samplers
    num_minibatches=2,
    update_epochs=2,

    gamma=0.85,
    gae_lambda=0.9,
    ent_coef=0.05,
    clip_coef=0.5,
    lr_schedule=dict(mode="const", start=0.0001),
    norm_adv=True,
    clip_vloss=True,
    max_grad_norm=1,
    weight_decay=0.05,

    rollouts_per_log=10,
    loglevel="DEBUG",
    run_name=None,
    trial_id=None,
    wandb_project="vcmi-gym",
    notes=None,
    vsteps_total=0,
    seconds_total=0,
    success_rate_target=None,
    ep_rew_mean_target=None,
    quit_on_target=False,
    mapside="defender",
    permasave_every=int(2e9),  # disable with int(2e9), which is always > time.time()
    out_dir_template="data/{group_id}",
    opponent_load_file=None,
    opponent_sbm_probs=[1, 0, 0],
    target_kl=None,
    logparams={},
    cfg_file=None,
    seed=42,
    skip_wandb_init=False,
    skip_wandb_log_code=False,
    envmaps=["gym/generated/4096/4x1024.vmap"],
    env=dict(
        random_stack_chance=0,
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
        user_timeout=600,
        vcmi_timeout=600,
        boot_timeout=300,
    ),
    # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
    env_wrappers=[dict(module="vcmi_gym.envs.util.wrappers", cls="LegacyObservationSpaceWrapper")],
    env_version=12,
    i2a_kwargs=dict(
        i2a_fc_units=1024,
        num_trajectories=10,
        rollout_dim=1024,
        rollout_policy_fc_units=1024,
        horizon=3,
        obs_processor_output_size=2048,
        transition_model_file="hauzybxn-model.pt",
        action_prediction_model_file="ogyesvkb-model.pt",
        reward_prediction_model_file="aexhrgez-model.pt",
    )
)
