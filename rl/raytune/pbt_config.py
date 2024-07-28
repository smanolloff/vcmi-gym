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

        # """
        # *** Resource allocation ***
        #
        # Instead of using ray.init(num_cpus=..., num_gpus=...)
        # and assigning per-worker requested resources CPU=..., GPU=...,
        # all workers request 0.01 CPU and 0.01 GPU (if available)
        #
        # => limit on spawned workers is defined by population_size
        #
        # Care must be taken to ensure this population has enough resources.
        # Measuring resource consumption with 1 MPPO process:
        #
        #       $ NO_WANDB=true NO_SAVE=true python -m rl.algos.mppo.mppo
        #
        # with config: num_envs=1, num_steps=256, update_epochs=10
        #
        # Results taken from `htop` with a huge refresh interfal (10+s)
        # to ensure multiple rollouts occur in a single refresh.
        # `htop` must be configured to show combined CPU usage in 1 bar
        # For GPU, `nvidia-smi` is used to obtain memory consumption.
        #
        #   mac: TODO
        #       16% of all (10) CPUs, spiking to 23%
        #       5% of all (16G) system memory
        #       ---
        #       => 6 workers for 96% of CPU (real CPU reported: 45%...)
        #
        #   pc (cpu):
        #       50.5% of all (8) CPUs
        #       14% (1.1G) of all (8G) system memory
        #       ---
        #       => not good for PBT (2 workers max)
        #
        #   pc (cuda):
        #       13% of all (8) CPUs
        #       14% (1.1G) of all (8G) system memory
        #       12.5% (500M) of all (8G) system memory
        #       ---
        #       => 6 workers for 78% of CPU, 75% of GPU, 84% of memory
        #
        #   dancho-server: TODO
        #
        #   mila-pc (cuda)
        #       1.7% of all (64) CPUs
        #       2% (1.2G) of all (64G) system memory
        #       8.5% (500M) of all (6.1G) GPU memory
        #       ---
        #       => 12 workers for 80% of GPU
        #
        #   mila-pc (cpu)
        #       18% of all (64) CPUs, spiking to 32%
        #       1.1% (700M) of all (64G) system memory
        #       => 4 workers for 72% of all CPU (real CPU reported: 7%...)
        #       => ...empirically found 16 CPU + 16 GPU workers is best
        #
        # """
        "population_size": 5,
        "cuda": True,  # use CUDA if available

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
    "tags": ["Map-v3", "StupidAI", "obstacles-random", "v3"],
    "mapside": "defender",  # attacker/defender; irrelevant if env.swap_sides > 0
    "envmaps": [
        # "gym/generated/4096/4096-mixstack-300K-01.vmap",
        # "gym/generated/4096/4096-mixstack-100K-01.vmap",
        # "gym/generated/4096/4096-mixstack-5K-01.vmap"
        "gym/generated/4096/4096-v3-100K-mod.vmap",
    ],
    "opponent_sbm_probs": [1, 0, 0],
    "opponent_load_file": None,
    # "opponent_load_file": "rl/models/Attacker model:v7/jit-agent.pt",

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
        "features_extractor1_stacks": [
            # => (B, 1960)
            {"t": "Unflatten", "dim": 1, "unflattened_size": [1, 20*98]},
            # => (B, 1, 1960)
            {"t": "Conv1d", "in_channels": 1, "out_channels": 8, "kernel_size": 98, "stride": 98},
            {"t": "Flatten"},
            {"t": "LeakyReLU"},
            # => (B, 160)
        ],
        "features_extractor1_hexes": [
            # => (B, 10725)
            {"t": "Unflatten", "dim": 1, "unflattened_size": [1, 165*65]},
            # => (B, 1, 10725)
            {"t": "Conv1d", "in_channels": 1, "out_channels": 4, "kernel_size": 65, "stride": 65},
            {"t": "Flatten"},
            {"t": "LeakyReLU"},
            # => (B, 660)
        ],
        "features_extractor2": [
            # => (B, 820)
            {"t": "Linear", "in_features": 820, "out_features": 512},
            {"t": "LeakyReLU"},
        ],
        "actor": {"t": "Linear", "in_features": 512, "out_features": 2312},
        "critic": {"t": "Linear", "in_features": 512, "out_features": 1}
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
    "rollouts_per_log": 50,
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
        "reward_dmg_factor": 5,
        "step_reward_fixed": -100,
        "step_reward_mult": 1,
        "term_reward_mult": 0,
        "reward_clip_tanh_army_frac": 1,
        "reward_army_value_ref": 500,
        "random_heroes": 1,
        "random_obstacles": 1,
        "town_chance": 10,
        "warmachine_chance": 40,
        "mana_min": 0,
        "mana_max": 100,
        "swap_sides": 0,
        "user_timeout": 60,
        "vcmi_timeout": 60,
        "boot_timeout": 300,
        "true_rng": False,
    },
    "seed": 0,
    "env_version": 3,
    "env_wrappers": [],
    # Wandb already initialized when algo is invoked
    # "run_id": None
    # "group_id": None
    # "run_name": None
    # "wandb_project": None
}
