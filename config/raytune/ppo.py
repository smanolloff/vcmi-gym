# https://docs.ray.io/en/latest/tune/api/search_space.html
config = {
    "wandb_project": "vcmi",
    "results_dir": "data",
    "population_size": 6,

    # Initial checkpoint to start from
    # "initial_checkpoint": "data/M4-PPO-20231125_020041/ad8ef_00002/checkpoint_000149/model.zip",
    "initial_checkpoint": None,

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
    "rollouts_per_iteration": 100,

    #
    # Number of logs per iteration
    # Requirement is: rollouts_per_iteration % logs_per_iteration == 0
    #
    "logs_per_iteration": 10,

    # """
    # Parameters are transferred from the top quantile_fraction
    # fraction of trials to the bottom quantile_fraction fraction.
    # Needs to be between 0 and 0.5.
    # Setting it to 0 essentially implies doing no exploitation at all.
    # """
    "quantile_fraction": 0.25,

    # NOTE:
    # Each of those params will be set in param_space
    # with a tune.uniform(min, max) automatically
    "hyperparam_bounds": {
        "learner_kwargs": {
            "learning_rate": [0.00001, 0.001],
            "gamma": [0.9, 0.999],
            "ent_coef": [0, 0.005],
        },
    },
    "param_space": {
        "learner_kwargs": {
            "policy": "MlpPolicy",
            "stats_window_size": 250,
            "learning_rate": 0.00001,
            "use_sde": False,
            "sde_sample_freq": -1,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.999,
            "gae_lambda": 0.98,
            "clip_range": 0.4,
            "normalize_advantage": True,
            "ent_coef": 0.001,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "env_kwargs": {
            "mapname": "ai/M2.vmap",
            "max_steps": 500,
            "vcmi_loglevel_global": "error",
            "vcmi_loglevel_ai": "error",
            "vcmienv_loglevel": "WARN",
            "consecutive_error_reward_factor": -1,
            "sparse_info": True,
        }
    }
}
