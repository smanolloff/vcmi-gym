# from ray.tune.search.sample import Integer, Float

N_WORKERS = 3
N_ENVS = 6

# overwrites learner_kwargs.n_steps to n_global_steps_max // n_envs
# (eg. 2048/48 = 42.666... => n_steps=42)
# N_GLOBAL_STEPS_MAX = 2048
N_GLOBAL_STEPS_MAX = 2000

# https://docs.ray.io/en/latest/tune/api/search_space.html
config = {
    "wandb_project": "vcmi",
    "results_dir": "data",
    "population_size": N_WORKERS,
    "target_ep_rew_mean": 300000,  # impossible target - 300k is the army value

    # Initial checkpoint to start from
    # "initial_checkpoint": "/Users/simo/Projects/vcmi-gym/data/GEN-PBT-MPPO-20240107_021740/2d08a_00000/checkpoint_000006/model.zip",  # noqa: E501
    "initial_checkpoint": None,

    # Perturb every N iterations
    "perturbation_interval": 1,

    #
    # Duration in rollouts of 1 training iteration
    #
    # One rollout is: 512 steps (learner_kwargs.n_steps) and takes ~0.8s
    # Episode length is: 25 steps max
    # => 1 rollout contains min. 20 episodes
    #
    # HOW TO CHOOSE:
    #   such that there at least 100 episodes between perturbations
    #

    "rollouts_per_iteration_step": 20,
    "rollouts_per_log": 2,

    "iteration_steps": 5,

    "hyperparam_mutations": {
        # "net_arch": [[], [64, 64], [256, 256]],
        "learner_kwargs": {
            "learning_rate": [0.00001, 0.0001, 0.0005],
            "gamma": [0.8, 0.9],
            # "batch_size": Integer(32, 256),  # breaks loading from file
            # "n_epochs": [5, 10, 20],
            # "gae_lambda": [0.95, 0.98],
            # "clip_range": [0.2, 0.5],
            # "vf_coef": [0.2, 0.5],
            # "max_grad_norm": [0.5, 1.5, 3],
            # "ent_coef": [0.01, 0.01],
            # "n_steps": [128, 256, 512, 1024, 2048, 4096, 8192],
        },
        # "optimizer": {"kwargs": {"weight_decay": [0, 0.01, 0.1]}},
    },

    # """
    # Parameters are transferred from the top quantile_fraction
    # fraction of trials to the bottom quantile_fraction fraction.
    # Needs to be between 0 and 0.5.
    # Setting it to 0 essentially implies doing no exploitation at all.
    # """
    "quantile_fraction": 0.25,
    "n_envs": N_ENVS,

    "all_params": {
        "learner_kwargs": {
            "stats_window_size": 100,
            "learning_rate": 0.00126,
            "n_steps": N_GLOBAL_STEPS_MAX // N_ENVS,
            # "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.8425,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "normalize_advantage": True,
            "ent_coef": 0.007,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            # "use_sde": False,  # n/a in MaskablePPO
        },
        "features_extractor_load_file": None,
        "features_extractor_load_file_type": None,  # model / params / sb3
        "features_extractor_freeze": False,
        "optimizer": {"class_name": "AdamW", "kwargs": {"eps": 1e-5, "weight_decay": 0}},
        "activation": "LeakyReLU",
        "net_arch": [],
        # "net_arch": [64, 64],
        "features_extractor": {
            "class_name": "VcmiFeaturesExtractor",
            "kwargs": {
                "layers": [
                    {"t": "Flatten"},
                    {"t": "Unflatten", "dim": 1, "unflattened_size": [165, 15]},
                    {"t": "VcmiAttention", "embed_dim": 15, "num_heads": 5, "batch_first": True},
                    {"t": "Flatten"},
                    {"t": "Linear", "in_features": 2475, "out_features": 256},
                    {"t": "LeakyReLU"},
                    {"t": "Linear", "in_features": 256, "out_features": 256},
                    {"t": "LeakyReLU"}
                ]
            }
        },
        "env_kwargs": {
            "max_steps": 1000,  # not used with MPPO
            "reward_dmg_factor": 5,
            "vcmi_loglevel_global": "error",
            "vcmi_loglevel_ai": "error",
            "vcmienv_loglevel": "WARN",
            "consecutive_error_reward_factor": -1,  # not used with MPPO
            "sparse_info": True,

            # Dynamically changed during training
            # "mapname": "ai/generated/A01.vmap",
            # "attacker": "MMAI_USER",
            # "defender": "StupidAI"
        },
        "mapmask": "ai/generated/B*.vmap",
        "randomize_maps": False,
    }
}
