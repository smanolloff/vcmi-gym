from ray.tune.search.sample import Integer, Float

# https://docs.ray.io/en/latest/tune/api/search_space.html
config = {
    "wandb_project": "vcmi",
    "results_dir": "data",
    "population_size": 1,

    # Initial checkpoint to start from
    # "initial_checkpoint": None,
    # "initial_checkpoint": "data/M6-PBT-PPO-20231128_000849/8c180_00004/checkpoint_000020/model.zip",
    # "enemy_checkpoint": "data/M6-PBT-PPO-20231128_000849/8c180_00004/checkpoint_000020/model.zip",

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
    "rollouts_per_iteration": 500,

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
            "max_steps": 1000,
            "vcmi_loglevel_global": "error",
            "vcmi_loglevel_ai": "error",
            "vcmienv_loglevel": "WARN",
            "consecutive_error_reward_factor": -1,
            "sparse_info": True,

            # Dynamically changed during training
            # "mapname": "ai/generated/A01.vmap",
            # "attacker": "MMAI_USER",
            # "defender": "StupidAI"
        },
        "map_pool": [
            "A01.vmap", "A02.vmap", "A03.vmap", "A04.vmap", "A05.vmap",
            "A06.vmap", "A07.vmap", "A08.vmap", "A09.vmap", "A10.vmap",
            "A11.vmap", "A12.vmap", "A13.vmap", "A14.vmap", "A15.vmap",
            "A16.vmap", "A17.vmap", "A18.vmap", "A19.vmap", "A20.vmap",
            "A21.vmap", "A22.vmap", "A23.vmap", "A24.vmap", "A25.vmap",
            "A26.vmap", "A27.vmap", "A28.vmap", "A29.vmap", "A30.vmap",
            "A31.vmap", "A32.vmap", "A33.vmap", "A34.vmap", "A35.vmap",
            "A36.vmap", "A37.vmap", "A38.vmap", "A39.vmap", "A40.vmap",
            "A41.vmap", "A42.vmap", "A43.vmap", "A44.vmap", "A45.vmap",
            "A46.vmap", "A47.vmap", "A48.vmap", "A49.vmap", "A50.vmap",
            "A51.vmap", "A52.vmap", "A53.vmap", "A54.vmap", "A55.vmap",
            "A56.vmap", "A57.vmap", "A58.vmap", "A59.vmap", "A60.vmap",
            "A61.vmap", "A62.vmap", "A63.vmap", "A64.vmap", "A65.vmap",
            "A66.vmap", "A67.vmap", "A68.vmap", "A69.vmap", "A70.vmap",
            "A71.vmap", "A72.vmap", "A73.vmap", "A74.vmap", "A75.vmap",
            "A76.vmap", "A77.vmap", "A78.vmap", "A79.vmap", "A80.vmap",
            "A81.vmap", "A82.vmap", "A83.vmap", "A84.vmap", "A85.vmap",
            "A86.vmap", "A87.vmap", "A88.vmap", "A89.vmap", "A90.vmap",
            "A91.vmap", "A92.vmap", "A93.vmap", "A94.vmap", "A95.vmap",
            "A96.vmap", "A97.vmap", "A98.vmap", "A99.vmap"
        ]
    }
}
