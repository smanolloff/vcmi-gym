# Test code from
# https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py

# Run with:
#
#   python -m rl.ray.main
#

from ray.tune.registry import register_env
from vcmi_gym import VcmiEnv_v4
from .mppo import MPPO_Config


#
# XXX: this directly initializes an Algorithm instance.
# However, with PBT it will be typically initialized by ray Tune.
# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html
#

config = MPPO_Config()

#
# AlgorithmConfig:
# https://docs.ray.io/en/releases-2.35.0/rllib/package_ref/algorithm.html#configuration-methods
#
# HOWEVER, docs are not up-to-date it seems.
# Best to look at the __init__ source, e.g. for environment:
# https://github.com/ray-project/ray/blob/ray-2.37.0/rllib/algorithms/algorithm_config.py#L324-L337
#

config.resources(
    placement_strategy="PACK",
    num_gpus=0,  # @OldAPIStack
    _fake_gpus=False,  # @OldAPIStack
    num_cpus_for_main_process=1,
)

config.framework(
    framework="torch",
)

config.api_stack(
    enable_rl_module_and_learner=True,
    enable_env_runner_and_connector_v2=True
)

config.environment(
    env="VCMI",
    env_config={"mapname": "gym/A1.vmap"},
    observation_space=None,     # inferred?
    action_space=None,          # inferred?
    clip_rewards=None,
    # normalize_actions=True,       # not used for Discrete action space
    # clip_actions=False,           # not used for Discrete action space
    # disable_env_checking=False,   # should be for MultiAgentEnv-only
)

config.env_runners(
    num_env_runners=0,          # 0=sample in main process
    num_envs_per_env_runner=1,  # i.e. vec_env.num_envs
    num_cpus_per_env_runner=1,
    num_gpus_per_env_runner=0,
    custom_resources_per_env_runner={},
    validate_env_runners_after_construction=True,
    sample_timeout_s=30.0,
    add_default_connectors_to_env_to_module_pipeline=True,
    add_default_connectors_to_module_to_env_pipeline=True,
    episode_lookback_horizon=1,
    rollout_fragment_length="auto",
    batch_mode="truncate_episodes",
    explore=False,
    compress_observations=False,
)

config.learners(
    num_learners=0,             # 0=learn in main process
    num_gpus_per_learner=0,
    num_cpus_per_learner=1,
)

config.training(
    gamma=0.8,
    lr=0.001,
    grad_clip=5,
    grad_clip_by="global_norm",           # global_norm = nn.utils.clip_grad_norm_(model.parameters)
    # train_batch_size_per_learner=None,  # for multi-learner setups
)

# config.callbacks()
# config.multi_agent()
# config.offline_data()

# XXX: local MPPO had 1 evaluation/hr (2~8K rollouts/hr)
# This is mostly due to training being much slower than experience collection
# (eval is vs. BattleAI).
# Switching to BattleAI will slow down training ~5-10x times.
#
# An evaluation (500 episodes) takes = ~30 min (on the PC).
#
# If evaluation_interval=1000:
#   For StupidAI training:
#   * new evaluation every 10..30 mins
#   * => up to 3 evaluators will run in parallel
#   For BattleAI training:
#   * new evaluation every 50..300 mins
#   * => up to 1 evaluators will run in parallel
#
# This is all based on local benchmarks, may be COMPLETELY different
# on distributed setups.
# (e.g. 10 workers running in parallel mean sigificantly faster rollouts)

# https://docs.ray.io/en/releases-2.35.0/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation.html
config.evaluation(
    evaluation_interval=500,    # training iterations (i.e. rollouts)
    evaluation_duration=100,
    evaluation_duration_unit="episodes",
    evaluation_sample_timeout_s=120.0,
    evaluation_parallel_to_training=False,  # rollouts are too quick for this
    evaluation_force_reset_envs_before_iteration=True,
    evaluation_config=MPPO_Config.overrides(
        explore=False,
        env_config=dict(mapname="gym/A2.vmap")
    ),
    evaluation_num_env_runners=2,

    # off_policy_estimation_methods={},     # relevant for offline observations
    # ope_split_batch_by_episode=True,      # relevant for offline observations
)

# This "reporting" seems to directly affect the training
# e.g. see this comment:
# https://github.com/ray-project/ray/blob/ray-2.37.0/rllib/algorithms/dreamerv3/dreamerv3.py#L136
config.reporting(
    metrics_num_episodes_for_smoothing=100,
    # metrics_episode_collection_timeout_s=60.0,  # seems used in v2 only? maybe now moved to c++?
    keep_per_episode_custom_metrics=False,

    # Repeat training_step() until some criteria is met
    min_time_s_per_iteration=None,
    min_train_timesteps_per_iteration=0,
    min_sample_timesteps_per_iteration=0,
)

config.checkpointing(
    export_native_model_files=False,            # will also .pt files in checkpoints
    checkpoint_trainable_policies_only=False,
)

# https://docs.ray.io/en/releases-2.35.0/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.debugging.html
# XXX: define a custom logger:
#      https://github.com/ray-project/ray/blob/master/python/ray/tune/logger/logger.py#L35
# config.debugging(
#     # logger_creator=None,
#     # logger_config=None,
#     log_level="WARN",
#     log_sys_usage=False,
#     seed=None,
# )

config.fault_tolerance(
    ignore_env_runner_failures=False,
    recreate_failed_env_runners=False,      # XXX: set to true for production
    max_num_env_runner_restarts=1000,
    delay_between_env_runner_restarts_s=10.0,
    restart_failed_sub_environments=False,
    num_consecutive_env_runner_failures_tolerance=100,
    env_runner_health_probe_timeout_s=30,
    env_runner_restore_timeout_s=1800,
)

config.rl_module(
    model_config_dict={
        "vf_share_layers": True,
        "obs_dims": {"misc": 4, "stacks": 2000, "hexes": 10725},
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
)

# XXX: config.build() starts the VCMI env
# => python error unless wrapped in __main__

if __name__ == "__main__":
    register_env("VCMI", lambda cfg: VcmiEnv_v4(**cfg))

    # Build a Algorithm object from the config and run 1 training iteration.
    algo = config.build()
    algo.train()
