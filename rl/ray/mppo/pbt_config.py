from .util import linlist, explist

pbt_config = {
    # "metric": "eval/is_success",
    "metric": "train/ep_rew_mean",
    "population_size": 2,
    "quantile_fraction": 0.5,
    "wandb_log_interval_s": 5,
    "training_step_duration_s": 10,
    "evaluation_episodes": 300,

    # XXX: this dict must match the AlgorithmConfig variables structure
    "hyperparam_mutations": {
        "entropy_coeff": linlist(0, 0.1, n=11),
        "clip_param": linlist(0.1, 0.7, n=7),
        "gamma": linlist(0.6, 0.999, n=13),
        "grad_clip": linlist(0.5, 10, n=11),

        # NOTE: train_batch_size MUST be divisible by:
        #       train_env_runners * rollout_fragment_length
        "train_batch_size_per_learner": [500, 1000, 2000, 4000],

        "use_kl_loss": [0, 1],
        "kl_coeff": linlist(0.01, 0.5, n=9),
        "kl_target": explist(0.001, 0.1, n=9),
        "lr": explist(1e-5, 2e-4, n=20),
        "lambda_": linlist(0.5, 0.99, n=20),
        "minibatch_size": [32, 64, 128],
        "num_epochs": linlist(1, 20, n=10, dtype=int),
        "vf_loss_coeff": linlist(0.1, 2, n=9),
        "vf_clip_param": linlist(0.1, 100, n=19),
        "env_config": {  # affects training env only
            "term_reward_mult": [0, 5]
        }
    },

    # Values below are used only if NOT present in hyperparam_mutations
    "clip_param": 0.3,
    "entropy_coeff": 0.0,
    "gamma": 0.8,
    "grad_clip": 5,
    "grad_clip_by": "global_norm",  # global_norm = nn.utils.clip_grad_norm_(model.parameters)
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    "lambda_": 1.0,
    "lr": 0.001,
    "minibatch_size": 20,
    "num_epochs": 1,
    "train_batch_size_per_learner": 100,
    "shuffle_batch_per_epoch": True,
    "use_critic": True,
    "use_gae": True,
    "use_kl_loss": True,
    "vf_clip_param": 10.0,
    "vf_loss_coeff": 1.0,

    "vf_share_layers": True,
    "network": {
        "attention": None,
        "features_extractor1_misc": [
            {"t": "LazyLinear", "out_features": 4},
            {"t": "LeakyReLU"},
        ],
        "features_extractor1_stacks": [
            {"t": "LazyLinear", "out_features": 8},
            {"t": "LeakyReLU"},
            {"t": "Flatten"},
        ],
        "features_extractor1_hexes": [
            {"t": "LazyLinear", "out_features": 8},
            {"t": "LeakyReLU"},
            {"t": "Flatten"},
        ],
        "features_extractor2": [
            {"t": "Linear", "in_features": 1484, "out_features": 512},
            {"t": "LeakyReLU"},
        ],
        "actor": {"t": "Linear", "in_features": 512, "out_features": 2312},
        "critic": {"t": "Linear", "in_features": 512, "out_features": 1}
    },
    "env": {
        "cls": "VcmiEnv_v4",
        "cpu_demand": {
            # As measured on Mac M1.
            # XXX: will probably differ on other CPUs. Might need a matrix.
            "BattleAI": 1,
            "StupidAI": 0.6
        },
        "episode_duration_s": {
            # Needed for properly setting "evaluation_sample_timeout_s"
            # Measured on Mac M1 during sampling
            # XXX: will probably differ on other CPUs. Might need a matrix.
            "BattleAI": 1.5,
            "StupidAI": 0.04
        },
        "step_duration_s": {
            # Needed for properly setting "rollout_fragment_length"
            # Measured on Mac M1 during sampling
            # XXX: will probably differ on other CPUs. Might need a matrix.
            "BattleAI": 0.08,
            "StupidAI": 0.002
        },
        "eval": {
            "runners": 2,  # 0=use main process
            "kwargs": {
                "mapname": "gym/generated/evaluation/8x64.vmap",
                "opponent": "BattleAI",
                "conntype": "thread"
            },
        },
        "train": {
            "runners": 2,  # 0=use main process
            "kwargs": {
                "mapname": "gym/generated/4096/4096-mixstack-100K-01.vmap",
                "opponent": "StupidAI",
                "reward_dmg_factor": 5,
                "step_reward_fixed": 0,
                "step_reward_frac": -0.001,
                "step_reward_mult": 1,
                "term_reward_mult": 0,
                "reward_clip_tanh_army_frac": 1,
                "reward_army_value_ref": 500,
                "reward_dynamic_scaling": False,
                "conntype": "thread"
            },
        },
        "common": {
            "kwargs": {
                "random_terrain_chance": 100,
                "town_chance": 10,
                "warmachine_chance": 40,
                "tight_formation_chance": 0,
                "battlefield_pattern": "",
                "random_heroes": 1,
                "random_obstacles": 1,
                "mana_min": 0,
                "mana_max": 0,
                "swap_sides": 0,
                "user_timeout": 60,
                "vcmi_timeout": 60,
                "boot_timeout": 300,
                "max_steps": 100,
                "sparse_info": True,
            },
        },
    },
    "experiment_name": "newray-test",  # overwritten via cmd-line args
    "wandb_project": "vcmi-gym",
}
