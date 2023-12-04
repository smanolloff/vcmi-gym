from ray.tune.search.sample import Integer, Float


# https://docs.ray.io/en/latest/tune/api/search_space.html
config = {
    "wandb_project": "vcmi",
    "results_dir": "data",
    "population_size": 6,

    # Initial checkpoint to start from
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
    "rollouts_per_iteration": 50,

    #
    # Number of logs per iteration
    # Requirement is: rollouts_per_iteration % logs_per_iteration == 0
    #
    "logs_per_iteration": 10,

    "hyperparam_mutations": {
        "learner_kwargs": {
            "learning_rate": Float(0.00001, 0.001),
            "gamma": Float(0.8, 0.999),
        },
    },

    # """
    # Parameters are transferred from the top quantile_fraction
    # fraction of trials to the bottom quantile_fraction fraction.
    # Needs to be between 0 and 0.5.
    # Setting it to 0 essentially implies doing no exploitation at all.
    # """
    "quantile_fraction": 0.25,

    "all_params": {
        "learner_kwargs": {
            "policy": "MlpPolicy",
            "stats_window_size": 100,
            "learning_rate": 0.0007,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.907,
            "gae_lambda": 0.98,
            "clip_range": 0.4,
            "normalize_advantage": True,
            "ent_coef": 0.007,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "env_kwargs": {
            "mapname": "ai/M8.vmap",
            "max_steps": 1000,
            "vcmi_loglevel_global": "error",
            "vcmi_loglevel_ai": "error",
            "vcmienv_loglevel": "WARN",
            "consecutive_error_reward_factor": -1,  # not used with MPPO
            "sparse_info": True,
        }
    }
}
