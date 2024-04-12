import logging

# https://docs.ray.io/en/latest/tune/api/search_space.html
config = {
    "wandb_project": "test",
    "results_dir": "data",
    "perturbation_interval": 1,
    "population_size": 1,

    # """
    # Parameters are transferred from the top quantile_fraction
    # fraction of trials to the bottom quantile_fraction fraction.
    # Needs to be between 0 and 0.5.
    # Setting it to 0 essentially implies doing no exploitation at all.
    # """
    "quantile_fraction": 0.25,

    # """
    # Perturbations will then be made based on the [min, max] values here
    # Initial values are sampled from here even if present in "all_params"
    # """
    "hyperparam_bounds": {
        "lr_schedule": {"start": [1e-7, 1e-4]},
        "ent_coef": [0, 0.2],
        "gae_lambda": [0.5, 0.99],
        "gamma": [0.6, 0.99],
        "max_grad_norm": [0.2, 10],
        "update_epochs": [2, 20],
        "vf_coef": [0.1, 2],
    },
    "all_params": {
        # Duration of one iteration
        #
        # Assuming num_steps=128:
        #   150K steps
        #   = 1171 rollouts
        #   = 6K episodes (good for 1K avg metric)
        #   = ~30..60 min (Mac)
        "vsteps_total": 1000,

        # Initial checkpoint to start from
        "agent_load_file": None,

        "tags": ["Map-3stack-01", "StupidAI", "encoding-float"],
        "mapside": "attacker",  # attacker/defender/both
        "mapmask": "gym/generated/88/88-3stack-30K-01.vmap",
        "opponent_sbm_probs": [1, 0, 0],
        "opponent_load_file": None,

        # PPO hyperparams
        "clip_coef": 0.4,
        "clip_vloss": False,
        "ent_coef": 0.007,
        "gae_lambda": 0.8,
        "gamma": 0.8425,
        "lr_schedule": {"mode": "const", "start": 0.00001},
        "max_grad_norm": 0.5,
        "norm_adv": True,
        "num_minibatches": 2,   # minibatch_size = rollout_buffer/num_minibatches,
        "num_steps": 128,       # rollout_buffer = num_steps*num_envs,
        "update_epochs": 10,    # full passes of rollout_buffer,
        "vf_coef": 1.2,
        "weight_decay": 0,

        # NN arch
        "network": {
            "features_extractor": [
                # => (B, 11, 15, 86|574)
                {"t": "Flatten", "start_dim": 2},
                {"t": "Unflatten", "dim": 1, "unflattened_size": [1, 11]},
                # => (B, 1, 11, 1290|8610)
                {"t": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": [1, 86], "stride": [1, 86], "padding": 0},
                # {"t": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": [1, 574], "stride": [1, 574], "padding": 0},
                {"t": "LeakyReLU"},
                # => (B, 32, 11, 15)
                {"t": "Flatten"},
                # => (B, 5280)
                {"t": "Linear", "in_features": 5280, "out_features": 1024},
                {"t": "LeakyReLU"},
                # => (B, 1024)
            ],
            "actor": {"t": "Linear", "in_features": 1024, "out_features": 2311},
            "critic": {"t": "Linear", "in_features": 1024, "out_features": 1}
        },

        "logparams": {
            "params/clip_coef": "clip_coef",
            "params/clip_vloss": "clip_vloss",
            "params/ent_coef": "ent_coef",
            "params/gae_lambda": "gae_lambda",
            "params/gamma": "gamma",
            # "params/lr_schedule": "lr_schedule",  # learning_rate is logged periodically
            "params/max_grad_norm": "max_grad_norm",
            "params/norm_adv": "norm_adv",
            "params/num_minibatches": "num_minibatches",
            "params/num_steps": "num_steps",
            "params/update_epochs": "update_epochs",
            "params/vf_coef": "vf_coef",
            "params/weight_decay": "weight_decay",
        },

        # Static
        "loglevel": logging.DEBUG,
        "skip_wandb_init": True,
        "skip_wandb_log_code": False,  # overwritten to True after 1st iteration
        "resume": False,
        "overwrite": [],
        "notes": "",
        "rollouts_per_mapchange": 0,
        "rollouts_per_log": 1,
        "rollouts_per_table_log": 0,
        "opponent_load_file": None,
        "success_rate_target": None,
        "ep_rew_mean_target": None,
        "quit_on_target": False,
        "randomize_maps": False,
        "save_every": 0,        # no effect (NO_SAVE=true)
        "permasave_every": 0,   # no effect (NO_SAVE=true)
        "max_saves": 3,         # no effect (NO_SAVE=true)
        "out_dir_template": "rl/data/{group_id}/{run_id}",  # relative to cwd
        "num_envs": 1,
        "env": {
            "encoding_type": "float",
            "reward_dmg_factor": 5,
            "step_reward_fixed": -100,
            "step_reward_mult": 1,
            "term_reward_mult": 0,
            "reward_clip_tanh_army_frac": 1,
            "reward_army_value_ref": 500,
            "random_combat": 1
        },
        "env_wrappers": [],
        # Set by the script:
        # "run_id": None
        # "group_id": None
        # "run_name": None
        # "wandb_project": None
    }
}
