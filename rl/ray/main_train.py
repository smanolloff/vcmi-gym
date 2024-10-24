# Test code from
# https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py

# Run with:
#
#   python -m rl.ray.main
#

from ray.tune.registry import register_env
from vcmi_gym import VcmiEnv_v4
from .mppo import MPPO_Config, MPPO_Callbacks

#
# XXX: this script directly initializes an Algorithm instance.
# However, with PBT it will be typically initialized by ray Tune.
# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html
#

config = MPPO_Config()

# Most of the `resources()` is moved to `env_runners()`
config.resources(
    num_cpus_for_main_process=1,
    placement_strategy="PACK",
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
    env_config=dict(mapname="gym/A1.vmap", sparse_info=True),

    action_mask_key="action_mask",  # not used as of ray-2.38.0
    action_space=None,              # inferred from env
    clip_actions=False,             #
    clip_rewards=None,              #
    disable_env_checking=False,     # one-time calls to step and reset
    is_atari=False,                 #
    normalize_actions=False,        # ignored for Discrete action space
    observation_space=None,         # inferred from env
    render_env=False,               #
)

config.env_runners(
    add_default_connectors_to_env_to_module_pipeline=True,
    add_default_connectors_to_module_to_env_pipeline=True,
    batch_mode="truncate_episodes",
    compress_observations=False,
    custom_resources_per_env_runner={},
    env_runner_cls=None,
    env_to_module_connector=None,
    episode_lookback_horizon=1,
    explore=False,
    max_requests_in_flight_per_env_runner=2,
    module_to_env_connector=None,
    num_cpus_per_env_runner=1,
    num_env_runners=0,          # 0 => sample in main process
    num_envs_per_env_runner=1,  # i.e. vec_env.num_envs
    num_gpus_per_env_runner=0,
    rollout_fragment_length="auto",
    update_worker_filter_stats=True,
    use_worker_filter_stats=True,
    sample_timeout_s=30.0,
    validate_env_runners_after_construction=True,
)

config.learners(
    num_learners=0,             # 0 => learn in main process
    num_gpus_per_learner=0,
    num_cpus_per_learner=1,
)

# !!! This is the API as of ray-2.38.0
# !!! It *will* change in future releases
config.training(
    clip_param=0.3,
    entropy_coeff=0.0,
    gamma=0.8,
    grad_clip=5,
    grad_clip_by="global_norm",       # global_norm = nn.utils.clip_grad_norm_(model.parameters)
    kl_coeff=0.2,
    kl_target=0.01,
    lambda_=1.0,
    lr=0.001,
    minibatch_size=3,
    num_epochs=1,
    train_batch_size_per_learner=6,  # i.e. batch_size; or mppo's num_steps when n_envs=1
    shuffle_batch_per_epoch=True,
    use_critic=True,
    use_gae=True,
    use_kl_loss=True,
    vf_clip_param=10.0,
    vf_loss_coeff=1.0,
)

config.callbacks(MPPO_Callbacks)

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

config.evaluation(
    evaluation_interval=0,      # in training iterations (i.e. rollouts)
    evaluation_duration=1,      # per evaluation env runner!
    evaluation_duration_unit="episodes",
    evaluation_sample_timeout_s=120.0,
    evaluation_parallel_to_training=False,  # training is too fast for this
    evaluation_force_reset_envs_before_iteration=True,
    evaluation_config=MPPO_Config.overrides(
        explore=False,
        env_config=dict(mapname="gym/A2.vmap", sparse_info=True)
    ),
    evaluation_num_env_runners=0,  # 0 => evaluate in main process

    # XXX: use this for testing siege/no siege/etc.
    #      (as well as custom W&B metrics?)
    custom_evaluation_function=None,

    # off_policy_estimation_methods={},     # relevant for offline observations
    # ope_split_batch_by_episode=True,      # relevant for offline observations
)

config.reporting(
    metrics_num_episodes_for_smoothing=100,
    keep_per_episode_custom_metrics=False,

    # XXX: seems used in old API only? maybe in v2 it's moved to c++?
    # metrics_episode_collection_timeout_s=60.0,

    # Repeat training_step() until some criteria is met
    # Use 0/none for 1 training_step() per 1 algorithm.step()
    # XXX: Metrics are broken for >1 training_step anyway:
    #      https://github.com/ray-project/ray/pull/48136
    # min_time_s_per_iteration=None,
    # min_train_timesteps_per_iteration=0,
    # min_sample_timesteps_per_iteration=0,
)

config.checkpointing(
    export_native_model_files=False,            # will also save .pt files in checkpoints
    checkpoint_trainable_policies_only=False,
)

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
    recreate_failed_env_runners=False,      # XXX: set to true for production
    ignore_env_runner_failures=False,
    max_num_env_runner_restarts=1000,
    delay_between_env_runner_restarts_s=10.0,
    restart_failed_sub_environments=False,
    num_consecutive_env_runner_failures_tolerance=100,
    env_runner_health_probe_timeout_s=30,
    env_runner_restore_timeout_s=1800,
)

config.rl_module(
    model_config={
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
    res = algo.train()
