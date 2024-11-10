import re
from dataclasses import dataclass, asdict

from . import util
from .common_env_runners import TrainEnv, EvalEnv
from .common_logger import Common_Logger

# ID for gym and tune envs must be different
# (the tune env has uses a custom env_creator with version validation)
# If IDs are the same, ray sometimes uses the default gym env creator
ENV_TUNE_ID = "VCMI_TUNE_ID"
ENV_VERSION_CHECK_KEY = "_version"


@dataclass
class UserConfig:
    # General
    env_gym_id: str
    env_runner_keepalive_interval_s: int
    experiment_name: str
    wandb_project: str
    wandb_log_interval_s: int

    # Tune
    hyperparam_mutations: dict
    hyperparam_values: dict
    metric: str
    population_size: int
    quantile_fraction: float
    training_step_duration_s: int

    # Other
    git_head: str
    git_is_dirty: bool
    master_overrides: dict
    model_load_file: str
    model_load_mapping: dict
    checkpoint_load_dir: str
    init_argument: str
    init_method: str
    timestamp: str

    # XXX: validations support primitive data types only
    def __post_init__(self):
        util.validate_dataclass_fields(self)

    def json_encode(self):
        return asdict(self)


def init(algo_config, cb):
    algo_config.enable_rl_module_and_learner = True
    algo_config.enable_env_runner_and_connector_v2 = True
    algo_config._master_config = None

    env_gym_id = "VCMI-v4"
    env_cls = util.get_env_cls(env_gym_id)
    env_cfg = {"conntype": "proc"}

    # Sanity-checking in restored experiments
    env_cfg[ENV_VERSION_CHECK_KEY] = env_cls.ENV_VERSION

    (
        # Default config
        algo_config
        .resources(
            num_cpus_for_main_process=1,
            placement_strategy="PACK")
        .framework(
            framework="torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True)
        .environment(
            env=ENV_TUNE_ID,
            env_config=env_cfg,
            action_mask_key="action_mask",  # not used as of ray-2.38.0
            action_space=None,
            clip_actions=False,
            clip_rewards=None,
            disable_env_checking=False,
            is_atari=False,
            normalize_actions=False,
            observation_space=None,
            render_env=False)
        .env_runners(
            add_default_connectors_to_env_to_module_pipeline=True,
            add_default_connectors_to_module_to_env_pipeline=True,
            batch_mode="truncate_episodes",
            compress_observations=False,
            custom_resources_per_env_runner={},
            env_runner_cls=TrainEnv,
            env_to_module_connector=None,
            episode_lookback_horizon=1,
            explore=False,
            max_requests_in_flight_per_env_runner=1,  # buggy, see https://github.com/ray-project/ray/pull/48499
            module_to_env_connector=None,
            num_env_runners=0,
            num_cpus_per_env_runner=0,  # see notes/ray_resources.txt
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=1,  # i.e. vec_env.num_envs
            rollout_fragment_length="auto",  # manually choosing a value is too complex
            update_worker_filter_stats=True,
            use_worker_filter_stats=True,
            sample_timeout_s=300,
            validate_env_runners_after_construction=True)
        .learners(
            num_learners=0,             # 0 => learn in main process
            num_gpus_per_learner=0,
            num_cpus_per_learner=1)
        .callbacks(cb)
        .multi_agent()
        .offline_data()
        .evaluation(
            evaluation_interval=1,  # !!! MUST BE 1
            evaluation_num_env_runners=0,
            evaluation_duration=100,
            evaluation_duration_unit="episodes",
            evaluation_sample_timeout_s=120.0,
            evaluation_parallel_to_training=False,
            evaluation_force_reset_envs_before_iteration=True,
            evaluation_config=dict(
                explore=False,
                env_config=env_cfg,
                num_cpus_per_env_runner=0,  # see notes/ray_resources.txt
                num_gpus_per_env_runner=0,
                num_envs_per_env_runner=1,
                env_runner_cls=EvalEnv),
            custom_evaluation_function=None)
        .reporting(
            metrics_num_episodes_for_smoothing=1000,   # auto-set (custom logic)
            keep_per_episode_custom_metrics=False,
            log_gradients=True,
            # metrics_episode_collection_timeout_s=60.0,  # seems old API
            min_time_s_per_iteration=None,
            min_train_timesteps_per_iteration=0,
            min_sample_timesteps_per_iteration=0)
        .checkpointing(
            export_native_model_files=False,    # not needed (have custom export)
            checkpoint_trainable_policies_only=False)
        .debugging(
            # Common_Logger currently logs nothing.
            # It's mostly used to supressing a deprecation warning.
            logger_config=dict(type=Common_Logger, prefix="Common_Logger_prefix"),
            log_level="DEBUG",
            log_sys_usage=False,
            seed=None)
        .fault_tolerance(
            recreate_failed_env_runners=True,  # useful for Amazon SPOT?
            ignore_env_runner_failures=False,
            max_num_env_runner_restarts=100,
            delay_between_env_runner_restarts_s=5.0,
            restart_failed_sub_environments=False,
            num_consecutive_env_runner_failures_tolerance=10,
            env_runner_health_probe_timeout_s=30,
            env_runner_restore_timeout_s=300)
        .rl_module(
            model_config=dict(
                env_version=env_cls.ENV_VERSION,
                network={
                    "attention": None,
                    "features_extractor1_misc": [{"t": "Flatten"}],
                    "features_extractor1_stacks": [{"t": "Flatten"}],
                    "features_extractor1_hexes": [{"t": "Flatten"}],
                    "features_extractor2": [{"t": "LazyLinear", "out_features": 64}],
                    "actor": {"t": "Linear", "in_features": 64, "out_features": 2312},
                    "critic": {"t": "Linear", "in_features": 64, "out_features": 1}
                },
                obs_dims={
                    "misc": env_cls.STATE_SIZE_MISC,
                    "stacks": env_cls.STATE_SIZE_STACKS,
                    "hexes": env_cls.STATE_SIZE_HEXES,
                },
                vf_share_layers=True))
        .user(
            env_gym_id=env_gym_id,
            env_runner_keepalive_interval_s=15,
            experiment_name="unnamed-experiment",
            wandb_project="",
            wandb_log_interval_s=60,

            hyperparam_mutations={},
            hyperparam_values={},
            metric="train/ep_rew_mean",
            population_size=1,
            quantile_fraction=0.5,
            training_step_duration_s=3600,

            git_head="",
            git_is_dirty=False,
            master_overrides={},
            model_load_file="",
            model_load_mapping={
                # rlmodule-to-model layer mapping
                "encoder.encoder": "encoder_actor",
                "pi": "actor",
                "vf": "critic",
            },
            checkpoint_load_dir="",
            init_argument="default",
            init_method="default",
            timestamp="2000-01-01T00:00:00")
    )


def configure_user(algo_config, kwargs):
    algo_config.user_config = UserConfig(**kwargs)
    return algo_config


#
# Usage:
# mppo_config = MPPO_Config()
# mppo_config.master_config(
#   "resources": {...},         # instead of mppo_config.resources(...)
#   "environment": {...},       # instead of mppo_config.environment(...)
#   ...                         # ...etc
# )
#
def configure_master(algo_config, cfg):
    algo_config._master_config = cfg

    for k, v in cfg.items():
        getattr(algo_config, k)(**v)

    # Make sure all evaluated episodes fit into the metric window
    # (discarding them is a pure waste of resources)
    if isinstance(algo_config.evaluation_duration, int) and algo_config.evaluation_duration_unit == "episodes":
        algo_config.metrics_num_episodes_for_smoothing = algo_config.evaluation_duration
        # assert algo_config.metrics_num_episodes_for_smoothing >= algo_config.evaluation_duration, (
        #     f"{algo_config.metrics_num_episodes_for_smoothing} >= {algo_config.evaluation_duration}"
        # )

    validate(algo_config)
    return algo_config


def validate(algo_config):
    assert algo_config.evaluation_interval == 1, "Tune expects eval results on each iteration"

    uc = algo_config.user_config

    assert uc.training_step_duration_s >= 0
    assert uc.wandb_log_interval_s >= 0
    assert uc.wandb_log_interval_s <= uc.training_step_duration_s
    assert re.match(r"^[\w_-]+$", uc.experiment_name), uc.experiment_name

    if uc.wandb_project:
        assert re.match(r"^[\w_-]+$", uc.wandb_project), uc.wandb_project

    if algo_config.num_learners > 0:
        # We can't setup wandb via algo_config.learner_group.foreach_learner(...)
        # wandb login must be ensured on all remotes prior to ray start
        # (+I don't understand how multi-learner setup works yet)
        raise Exception("TODO(simo): wandb setup in remote learners is not implemented")

    def validate_hyperparam_mutations(mut):
        for k, v in mut.items():
            if isinstance(v, list):
                assert all(isinstance(v1, (int, float)) for v1 in v), (
                    f"hyperparam_mutations for (possibly nested) key {repr(k)} contains invalid value types"
                )
            elif isinstance(v, dict):
                validate_hyperparam_mutations(v)
            else:
                raise Exception(f"Invalid hyperparam value type for (possibly nested) key {repr(k)}")

    validate_hyperparam_mutations(algo_config.user_config.hyperparam_mutations or {})

    if algo_config.env_config["conntype"] == "thread" and algo_config.num_env_runners == 0:
        raise Exception("Train runners are local and cannot have conntype='thread'")

    if algo_config.evaluation_config["env_config"]["conntype"] == "thread" and algo_config.evaluation_num_env_runners == 0:
        raise Exception("Eval runners are local and cannot have conntype='thread'")
