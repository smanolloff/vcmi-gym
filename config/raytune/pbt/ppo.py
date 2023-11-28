from ray.tune.search.sample import Integer, Float


# https://docs.ray.io/en/latest/tune/api/search_space.html
config = {
    "wandb_project": "vcmi",
    "results_dir": "data",
    "population_size": 6,

    # Initial checkpoint to start from
    # "initial_checkpoint": None,
    "initial_checkpoint": "data/M6-PBT-PPO-20231128_000849/8c180_00004/checkpoint_000020/model.zip",

    # Perturb every N iterations
    "perturbation_interval": 1,

    #
    # Duration in rollouts of 1 training iteration
    #
    # One rollout is: 512 steps (learner_kwargs.n_steps) and takes ~0.8s
    # Episode length is: ~400 steps (beginning)...~75 steps (end)
    #
    # HOW TO CHOOSE:
    #   such that there at least 100 episodes between perturbations
    #   (if perturbation_interval=1, choose rollouts_per_iteration > 100)
    #
    "rollouts_per_iteration": 1000,

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
        "env_kwargs": {
          "consecutive_error_reward_factor": Integer(-1000, -1)
        }
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
            "use_sde": False,
            "sde_sample_freq": -1,
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
            "mapname": "ai/M7.vmap",
            "max_steps": 5000,
            "vcmi_loglevel_global": "error",
            "vcmi_loglevel_ai": "error",
            "vcmienv_loglevel": "WARN",
            "consecutive_error_reward_factor": -1,
            "sparse_info": True,
        }
    }
}
