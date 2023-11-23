# https://docs.ray.io/en/latest/tune/api/search_space.html
config = {
    "wandb_project": "vcmi",
    "results_dir": "data",
    "population_size": 6,

    # Training speed is:
    #   ~2 rollouts/s (beginning, with many invalid actions)
    #   ~1.3 rollouts/s (end)
    #
    # HOW TO CHOOSE:
    #   such that perturbation is performed once every 10? minutes
    #
    # Example:
    #   1000 ~= 500..769 seconds/pertrubation
    #
    "perturbation_interval": 500,

    # Reduce ray reporting to once every X rollouts
    # (equivalent of learner_kwargs.log_interval for tensorboard)
    #
    # Episode length is:
    #   ~400 steps (beginning)
    #   ~75 steps (end)
    # Rollout size is 512 steps (learner_kwargs.n_steps), ie.:
    #   1.25 episodes (beginning)
    #   7 episodes (end)
    #
    # HOW TO CHOOSE:
    #   such that there at least 100 episodes between reports
    #
    # Example:
    #   50 ~= 350..62 episodes/report
    #   20 ~= 140..25 episodes/report
    #
    # NOTE: perturbation_interval will be recalculated accordingly
    "reduction_factor": 50,

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
            "gamma": [0.9, 0.999],
            "ent_coef": [0, 0.005],
        },
        "learning_rate": [0.00001, 0.001],
    },
    "param_space": {
        "model_load_file": None,
        "model_load_update": True,
        "progress_bar": False,
        "reset_num_timesteps": False,
        # this is added programattically based on top-level "results_dir" param
        # "out_dir_template": "data/{experiment_name}/{trial_id}",
        "log_tensorboard": True,
        "total_timesteps": 10e6,
        "n_checkpoints": 0,  # no saves
        "learner_kwargs": {
            "policy": "MlpPolicy",
            "stats_window_size": 250,
            "use_sde": False,
            "sde_sample_freq": -1,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.999,
            "gae_lambda": 0.98,
            "clip_range": 0.4,
            "normalize_advantage": True,
            "ent_coef": 0.0001,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        # this is added programatically based on top-level "reduction_factor"
        # "log_interval": 25
        "learner_lr_schedule": None,
        "learning_rate": 0.00001,
        "env_kwargs": {
            "mapname": "AI-1.vmap",
            "max_steps": 500,
            "vcmi_loglevel_global": "error",
            "vcmi_loglevel_ai": "error",
            "vcmienv_loglevel": "WARN",
            "consecutive_error_reward_factor": -1,
            "sparse_info": True,
        }
    }
}
