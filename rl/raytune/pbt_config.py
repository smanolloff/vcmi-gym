import logging
from .common import linlist, explist

# https://docs.ray.io/en/latest/tune/api/search_space.html
config = {
    "_raytune": {
        "target_ep_rew_mean": 3_000_000,  # impossible target
        "wandb_project": "vcmi-gym",
        "perturbation_interval": 1,
        "synch": True,
        "time_attr": "training_iteration",  # XXX: don't use time_total_s

        "population_size": 6,

        # """
        # Parameters are transferred from the top quantile_fraction
        # fraction of trials to the bottom quantile_fraction fraction.
        # Needs to be between 0 and 0.5.
        # Setting it to 0 essentially implies doing no exploitation at all.
        # """
        "quantile_fraction": 0.25,

        # """
        # Perturbations will then be made based on the values here
        #
        # XXX: Do NOT use tune distributions here (e.g. tune.uniform()).
        #      Reasons why this is bad:
        #   > tune will perturb by simply multiplying the value by 0.8 or 1.2
        #       e.g. range 0.95-0.99, old=0.96 means new value of 0.96*1.2 = 1.152
        #   > tune will ignore the value bounds which can lead to errors
        #   > tune will treat everything as a continuous distribution
        #       e.g. tune.choice([1, 2]), perturbation will still perturb as above
        #
        # Workaround: use *plain lists* with appropriately distributed elements:
        #   > tune will perturb by choosing the next or prev value in the list
        #   > tune will not honor the data type (values are converted to floats)
        #       e.g. [1, 2] will pass in 1.0 or 2.0 -- this must be accounted for
        # """
        "hyperparam_mutations": {
            "ent_coef": linlist(0, 0.1, n=11),
            "gamma": linlist(0.6, 0.999, n=13),
            "max_grad_norm": linlist(0.5, 10, n=11),
            "num_steps": [128, 256, 512],

            # PPO-vanilla specific
            "lr_schedule": {"start": explist(5e-7, 1e-4, n=20)},
            "gae_lambda": linlist(0.5, 0.99, n=20),
            "num_minibatches": [2, 4, 8],
            "update_epochs": linlist(2, 20, n=5, dtype=int),
            "vf_coef": linlist(0.1, 2, n=9),

            # PPO-DNA specific
            # "lr_schedule_value": {"start": explist(1e-7, 1e-4, n=20)},
            # "lr_schedule_policy": {"start": explist(1e-7, 1e-4, n=20)},
            # "lr_schedule_distill": {"start": explist(1e-7, 1e-4, n=20)},
            # "num_minibatches_distill": [2, 4, 8],
            # "num_minibatches_policy": [2, 4, 8],
            # "num_minibatches_value": [2, 4, 8],
            # "update_epochs_distill": linlist(2, 10, n=5, dtype=int),
            # "update_epochs_policy": linlist(2, 10, n=5, dtype=int),
            # "update_epochs_value": linlist(2, 10, n=5, dtype=int),
            # "gae_lambda_policy": linlist(0.59, 0.99, n=9),
            # "gae_lambda_value": linlist(0.59, 0.99, n=9),
            # "distill_beta": linlist(0.5, 1.0, n=6),
        },

        #
        # XXX: The values will be used once again when RESUMING an experiment:
        #       Trials will first be resumed with the values they had during
        #       the interruption, but the first trial to finish will get
        #       a set of "initial" values for its next iteration.
        #
        "initial_hyperparams": {
            # "lr_schedule": {"mode": "const", "start": 1.1e-5},
            # "ent_coef": 0.02,
            # "gae_lambda": 0.95,
            # "gamma": 0.99,
            # "max_grad_norm": 6,
            # "num_minibatches": 2,
            # "num_steps": 128,
            # "update_epochs": 10,
            # "vf_coef": 1.2,
        },

        "resumes": [],  # trial_id will be appended here when resuming (trial_id != run_id)
        "resumed_run_id": None,  # will be set to the *original* run_id to resume
    },

    #
    # Algo params
    #

    # Duration of one iteration
    #
    # Assuming num_steps=128:
    #   150K steps
    #   = 1171 rollouts
    #   = 6K episodes (good for 1K avg metric)
    #   = ~30..60 min (Mac)
    # "vsteps_total": 150_000,
    "seconds_total": 1800,

    # Initial checkpoint to start from
    # "agent_load_file": "rl/models/model-PBT-mppo-attacker-cont-20240517_134625.b6623_00000:v2/agent.pt",
    # "agent_load_file": "rl/models/model-PBT-mppo-defender-20240521_112358.79ad0_00000:v1/agent.pt",
    "agent_load_file": None,

    # "agent_load_file": None,
    "tags": ["Map-4096-mixstack", "StupidAI", "obstacles-random", "encoding-float"],
    "mapside": "attacker",  # attacker/defender; irrelevant if env.swap_sides > 0
    "envmaps": [
        "gym/generated/4096/4096-mixstack-300K-01.vmap",
        "gym/generated/4096/4096-mixstack-100K-01.vmap",
        "gym/generated/4096/4096-mixstack-5K-01.vmap"
    ],
    "opponent_sbm_probs": [1, 0, 0],
    "opponent_load_file": None,
    # "opponent_load_file": "rl/models/model-PBT-mppo-defender-20240521_112358.79ad0_00000:v1/jit-agent.pt",

    #
    # PPO hyperparams
    #
    # XXX: values here are used only if the param is NOT present
    #       in "hyperparam_mutations". For initial mutation values,
    #       use the "initial_hyperparams" dict.
    "clip_coef": 0.4,
    "clip_vloss": False,
    "ent_coef": 0.007,
    "gamma": 0.8425,
    "max_grad_norm": 0.5,
    "norm_adv": True,
    "num_steps": 128,       # rollout_buffer = num_steps*num_envs,
    "weight_decay": 0,

    # Vanilla PPO specific
    "lr_schedule": {"mode": "const", "start": 0.00001},
    "num_minibatches": 2,   # minibatch_size = rollout_buffer/num_minibatches,
    "update_epochs": 10,    # full passes of rollout_buffer,
    "gae_lambda": 0.8,
    "vf_coef": 1.2,

    # PPO-DNA specific
    # "lr_schedule_value": {"mode": "const", "start": 0.00001},
    # "lr_schedule_policy": {"mode": "const", "start": 0.00001},
    # "lr_schedule_distill": {"mode": "const", "start": 0.00001},
    # "num_minibatches_distill": 4,
    # "num_minibatches_policy": 4,
    # "num_minibatches_value": 4,
    # "update_epochs_distill": 4,
    # "update_epochs_policy": 2,
    # "update_epochs_value": 4,
    # "gae_lambda_policy": 0.7,
    # "gae_lambda_value": 0.8,
    # "distill_beta": 1.0,

    # NN arch
    "network": {
        "attention": None,
        "features_extractor": [
            # => (B, 11, 15, 86)
            {"t": "Flatten"},
            {"t": "Unflatten", "dim": 1, "unflattened_size": [1, 14355]},
            # => (B, 1, 14190)
            {"t": "Conv1d", "in_channels": 1, "out_channels": 32, "kernel_size": 87, "stride": 87},
            {"t": "LeakyReLU"},
            # => (B, 32, 165)
            {"t": "Flatten"},
            {"t": "Linear", "in_features": 5280, "out_features": 1024},
            {"t": "LeakyReLU"},
        ],
        "actor": {"t": "Linear", "in_features": 1024, "out_features": 2311},
        "critic": {"t": "Linear", "in_features": 1024, "out_features": 1}
    },


    # Static
    "loglevel": logging.INFO,
    "logparams": {},  # overwritten based on "hyperparam_mutations"
    "skip_wandb_init": True,
    "skip_wandb_log_code": False,  # overwritten to True after 1st iteration
    "trial_id": None,  # overwritten based on the trial id
    "resume": False,
    "overwrite": [],
    "notes": "",
    "rollouts_per_mapchange": 0,
    "rollouts_per_log": 1,
    "rollouts_per_table_log": 0,
    "success_rate_target": None,
    "ep_rew_mean_target": None,
    "quit_on_target": False,
    "save_every": 0,        # no effect (NO_SAVE=true)
    "permasave_every": 0,   # no effect (NO_SAVE=true)
    "max_saves": 3,         # no effect (NO_SAVE=true)
    "out_dir_template": "data/{group_id}/{run_id}",  # relative project root
    "num_envs": 1,
    "env": {
        "encoding_type": "float",
        "reward_dmg_factor": 5,
        "step_reward_fixed": -100,
        "step_reward_mult": 1,
        "term_reward_mult": 0,
        "reward_clip_tanh_army_frac": 1,
        "reward_army_value_ref": 500,
        "random_heroes": 1,
        "random_obstacles": 1,
        "swap_sides": 0,
        "user_timeout": 60,
        "vcmi_timeout": 60,
        "boot_timeout": 300,
        "seed": 0,  # random
    },
    "env_wrappers": [],
    # Wandb already initialized when algo is invoked
    # "run_id": None
    # "group_id": None
    # "run_name": None
    # "wandb_project": None
}
