from ..common import util


def load():
    # Cannot use "env_cls_name" because "VcmiEnv_v4" is *not* the cls name
    # e.g. vcmi_gym.get("VcmiEnv_v4").__name__ => "VcmiEnv"
    # => use gym registry
    # (assuming caller has invoked vcmi_gym.register_envs() already)

    env_gym_id = "VCMI-v4"
    env_cls = util.get_env_cls(env_gym_id)

    env_kwargs_common = {
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
        "user_timeout": 300,  # shutdown is slow during PBT pertubs
        "vcmi_timeout": 60,
        "boot_timeout": 300,
        "max_steps": 100,
        "sparse_info": True,
    }

    # Measured on Mac M2 during sampling
    # XXX: will probably differ on other CPUs. Might need a matrix.
    ai_stats = {
        "cpu_usage": {"StupidAI": 0.6, "BattleAI": 1},
        "step_duration_s": {"StupidAI": 0.002, "BattleAI": 0.08},
        "episode_duration_s": {"StupidAI": 0.04, "BattleAI": 1.5}
    }

    train_opponent = "StupidAI"
    train_env_runners = 2

    eval_opponent = "BattleAI"
    eval_env_runners = 2
    eval_episodes = 10

    # XXX: this dict must match the AlgorithmConfig variables structure...
    hyperparam_mutations = {
    }

    calculated_train_sample_timeout = util.calc_train_sample_timeout_s(
        hyperparam_mutations.get("train_batch_size_per_learner", [2000]),
        train_env_runners,
        ai_stats["step_duration_s"][train_opponent]
    )

    calculated_eval_sample_timeout = util.calc_eval_sample_timeout_s(
        eval_episodes,
        ai_stats["step_duration_s"][eval_opponent],
        100,
    )

    #
    # XXX: this dict's root keys are METHODS called on the MPPO_Config
    #      e.g. {"environment": {"clip_actions": False}}
    #      => cfg.environment(clip_actions=False)
    #
    return dict(
        # TRAIN runners
        environment={
            "env_config": util.deepmerge(env_kwargs_common, {
                "mapname": "gym/generated/4096/4096-mixstack-100K-01.vmap",
                "opponent": "StupidAI",
                "reward_dmg_factor": 5,
                "step_reward_fixed": 0,
                "step_reward_frac": -0.001,
                "step_reward_mult": 1,
                "term_reward_mult": 1,
                "reward_clip_tanh_army_frac": 1,
                "reward_army_value_ref": 500,
                "reward_dynamic_scaling": False,
                "conntype": "thread"
            }),
        },

        env_runners={
            "custom_resources_per_env_runner": {
                "train_cpu": ai_stats["cpu_usage"][train_opponent],
                "eval_cpu": 0,
            },
            "num_env_runners": train_env_runners,
            "sample_timeout_s": calculated_train_sample_timeout,
            "rollout_fragment_length": 100,
        },

        # EVAL runners
        evaluation={
            "evaluation_config": {
                "env_config": util.deepmerge(env_kwargs_common, {
                    "mapname": "gym/generated/evaluation/8x64.vmap",
                    "opponent": "BattleAI",
                    "conntype": "thread"
                }),
                "custom_resources_per_env_runner": {
                    "train_cpu": 0,
                    "eval_cpu": ai_stats["cpu_usage"][eval_opponent],
                },
                # "num_env_runners": eval_env_runners,  # not working
                # "sample_timeout_s": calculated_eval_sample_timeout  # not working
            },
            "evaluation_num_env_runners": eval_env_runners,
            "evaluation_duration": eval_episodes,
            "evaluation_sample_timeout_s": calculated_eval_sample_timeout,
            "evaluation_interval": 100,
        },

        rl_module={
            "model_config": {
                "env_version": env_cls.ENV_VERSION,
                "obs_dims": {
                    "misc": env_cls.STATE_SIZE_MISC,
                    "stacks": env_cls.STATE_SIZE_STACKS,
                    "hexes": env_cls.STATE_SIZE_HEXES,
                },
                "network": {
                    "attention": None,
                    "features_extractor1_misc": [
                        {"t": "Dense", "units": 4},
                        {"t": "ReLU", "negative_slope": 0.01},
                    ],
                    "features_extractor1_stacks": [
                        {"t": "Dense", "units": 8},
                        {"t": "ReLU", "negative_slope": 0.01},
                        {"t": "Flatten"},
                    ],
                    "features_extractor1_hexes": [
                        {"t": "Dense", "units": 8},
                        {"t": "ReLU", "negative_slope": 0.01},
                        {"t": "Flatten"},
                    ],
                    "features_extractor2": [
                        {"t": "Dense", "input_shape": [1484], "units": 512},
                        {"t": "ReLU", "negative_slope": 0.01},
                    ],
                    "actor": {"t": "Dense", "input_shape": [512], "units": 2312},
                    "critic": {"t": "Dense", "input_shape": [512], "units": 1}
                }
            }
        },

        # Fixed values to use in case a hyperparam is not mutable
        training={
            "model_size": "XS",
            "training_ratio": 32,
            "gc_frequency_train_steps": 100,
            "batch_size_B": 4,
            "batch_length_T": 8,
            "horizon_H": 15,
            "gae_lambda": 0.95,
            "entropy_scale": 3e-4,
            "return_normalization_decay": 0.99,
            "train_critic": True,
            "train_actor": True,
            "intrinsic_rewards_scale": 0.1,
            "world_model_lr": 1e-4,
            "actor_lr": 3e-5,
            "critic_lr": 3e-5,
            "world_model_grad_clip_by_global_norm": 1000.0,
            "critic_grad_clip_by_global_norm": 100.0,
            "actor_grad_clip_by_global_norm": 100.0,
            "symlog_obs": True,
            "use_float16": False,
        },

        user={
            #
            # General
            #
            "env_gym_id": env_gym_id,
            "env_runner_keepalive_interval_s": 15,  # smaller than VCMI timeouts
            "wandb_log_interval_s": 60,
            "model_load_file": "",
            "model_load_mapping": {
                "encoder.encoder": "encoder_actor",
                "pi": "actor",
                "vf": "critic",
            },

            #
            # Tune
            #
            "hyperparam_mutations": hyperparam_mutations,
            "hyperparam_values": {},
            "metric": "train/ep_rew_mean",
            "population_size": 4,
            "quantile_fraction": 0.4,
            "training_step_duration_s": 3600,

            #
            # Updated programatically, do NOT edit here
            #
            "checkpoint_load_dir": None,
            "experiment_name": None,      # --experiment-name
            "git_head": None,
            "git_is_dirty": None,
            "master_overrides": None,     # --overrides
            "init_argument": None,        # --init-argument
            "init_method": None,          # --init-method
            "timestamp": None,
            "wandb_project": None,        # --wandb-project
        },
    )
