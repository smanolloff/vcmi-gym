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
    train_env_runners = 1

    eval_opponent = "BattleAI"
    eval_env_runners = 1
    eval_episodes = 10

    # XXX: this dict must match the AlgorithmConfig variables structure...
    hyperparam_mutations = {
        # "vtrace_clip_rho_threshold": 1.0,
        # "vtrace_clip_pg_rho_threshold": 1.0,
        # "learner_queue_size": 3,
        # "max_requests_in_flight_per_aggregator_worker": 2,
        # "timeout_s_sampler_manager": 0.0,
        # "timeout_s_aggregator_manager": 0.0,
        # "broadcast_interval": 1,
        # "num_aggregation_workers": 0,
        "entropy_coeff": util.linlist(0, 0.1, n=11),
        "gamma": util.linlist(0.6, 0.999, n=13),
        "grad_clip": util.linlist(0.5, 10, n=11),
        "lr": util.explist(1e-5, 2e-4, n=20),
        "minibatch_size": [25, 50, 100],
        "num_epochs": util.linlist(1, 20, n=10, dtype=int),
        # XXX: 1 obs = 45K, 10K => 450M
        "train_batch_size_per_learner": [2000, 5000, 10000],
        "vf_loss_coeff": util.linlist(0.5, 1.5, n=11),
        # "env_config": {  # affects training env only
        #     "term_reward_mult": [0, 5]
        # }
    }

    calculated_train_sample_timeout = util.calc_train_sample_timeout_s(
        hyperparam_mutations.get("train_batch_size_per_learner", [5000]),
        train_env_runners,
        ai_stats["step_duration_s"][train_opponent]
    )

    calculated_eval_sample_timeout = util.calc_eval_sample_timeout_s(
        eval_episodes,
        eval_env_runners,
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
                "term_reward_mult": 0,
                "reward_clip_tanh_army_frac": 1,
                "reward_army_value_ref": 500,
                "reward_dynamic_scaling": False,
                "conntype": "proc"
            }),
        },

        env_runners={
            "custom_resources_per_env_runner": {
                "train_cpu": ai_stats["cpu_usage"][train_opponent],
                "eval_cpu": 0,
            },
            "max_requests_in_flight_per_env_runner": 3,
            "num_env_runners": train_env_runners,
            "sample_timeout_s": calculated_train_sample_timeout
        },

        # EVAL runners
        evaluation={
            "evaluation_config": {
                "env_config": util.deepmerge(env_kwargs_common, {
                    "mapname": "gym/generated/evaluation/8x64.vmap",
                    "opponent": "BattleAI",
                    "conntype": "proc"
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
            "evaluation_sample_timeout_s": calculated_eval_sample_timeout
        },

        rl_module={
            "model_config": {
                "env_version": env_cls.ENV_VERSION,
                "obs_dims": {
                    "misc": env_cls.STATE_SIZE_MISC,
                    "stacks": env_cls.STATE_SIZE_STACKS,
                    "hexes": env_cls.STATE_SIZE_HEXES,
                },
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
                }
            }
        },

        # Fixed values to use in case a hyperparam is not mutable
        training={
            "vtrace_clip_rho_threshold": 1.0,
            "vtrace_clip_pg_rho_threshold": 1.0,
            "learner_queue_size": 3,
            "max_requests_in_flight_per_aggregator_worker": 2,
            "broadcast_interval": 1,
            "num_aggregation_workers": 0,
            "entropy_coeff": 0.0,
            "gamma": 0.8,
            "grad_clip": 5,
            "grad_clip_by": "global_norm",  # global_norm = nn.utils.clip_grad_norm_(model.parameters)
            "lr": 0.001,
            "num_epochs": 1,
            "train_batch_size_per_learner": 500,
            "shuffle_batch_per_epoch": True,
            "vf_loss_coeff": 1.0,
        },

        user={
            #
            # General
            #
            "env_gym_id": env_gym_id,
            "env_runner_keepalive_interval_s": 15,  # smaller than VCMI timeouts
            "wandb_log_interval_s": 0,
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
            "population_size": 1,
            "quantile_fraction": 0.3,
            "training_step_duration_s": 15,

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
