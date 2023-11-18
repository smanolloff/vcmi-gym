from ray import tune

# https://docs.ray.io/en/latest/tune/api/search_space.html

config = {
    "wandb_project": "Wandb_example",

    # used for calculating perturbation_interval
    # based on total_timesteps and n_steps
    "desired_perturbations": 100,

    # """
    # Parameters are transferred from the top quantile_fraction
    # fraction of trials to the bottom quantile_fraction fraction.
    # Needs to be between 0 and 0.5.
    # Setting it to 0 essentially implies doing no exploitation at all.
    # """
    "quantile_fraction": 0.25,

    # NOTE:
    # """
    # Tune will sample uniformly between the bounds provided by
    # hyperparam_bounds for the initial hyperparameter values if
    # the corresponding hyperparameters are not present in a
    # trial’s initial config.
    # """
    # NOTE: apparently, only floats are supported here.
    #
    "hyperparam_bounds": {
        "learner_kwargs": {
            "gamma": [0.9, 0.99],
            "ent_coef": [0, 0.005],
        },
        "learning_rate": [0.00001, 0.0003],
        "env_kwargs": {
            "consecutive_error_reward_factor": [-5, -1],
        }
    },
    "param_space": {
        "model_load_file": None,
        "model_load_update": True,
        "progress_bar": False,
        "reset_num_timesteps": True,
        "out_dir_template": "data/PPO-{run_id}",
        "log_tensorboard": True,
        "total_timesteps": 3000e3,
        "max_episode_steps": 5000,
        "n_checkpoints": 0,  # no saves
        "learner_kwargs": {
            "policy": "MlpPolicy",
            "use_sde": False,
            "sde_sample_freq": -1,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            # "gamma": tune.uniform(0.9, 0.999),
            "gae_lambda": 0.98,
            "clip_range": 0.4,
            "normalize_advantage": True,
            # "ent_coef": tune.uniform(0, 0.001),
            "vf_coef": 0.5,
            "max_grad_norm": 0.5
        },
        "learner_lr_schedule": None,
        # "learning_rate": tune.uniform(0.00001, 0.0002),
        "env_kwargs": {
            "mapname": "AI-1.vmap",
            "vcmi_loglevel_global": "error",
            "vcmi_loglevel_ai": "warn",
            # "consecutive_error_reward_factor": tune.uniform(-1, -5),
        }
    }
}
