import ray.tune
import os
import vcmi_gym
from ray.rllib.utils import deep_update
from .pbt_config import pbt_config
from .util import deepmerge
from . import MPPO_Algorithm, MPPO_Config, MPPO_Callback, MPPO_Logger

env_cls = getattr(vcmi_gym, pbt_config["env"]["cls"])


def calculate_fragment_duration_s(cfg, train_env_cfg, eval_env_cfg):
    if "train_batch_size_per_learner" in cfg["hyperparam_mutations"]:
        batch_sizes = cfg["hyperparam_mutations"]["train_batch_size_per_learner"]
    else:
        batch_sizes = [cfg["train_batch_size_per_learner"]]

    opponent = train_env_cfg["kwargs"]["opponent"]
    max_fragment_length = max(batch_sizes) / max(1, train_env_cfg["runners"])
    step_duration_s = cfg["env"]["step_duration_s"][opponent]
    fragment_duration_s = max_fragment_length * step_duration_s
    print(f"Estimated time for collecting samples: {fragment_duration_s:.1f}s")

    # Maximum allowed time for sample collection (hard-coded)
    max_fragment_duration_s = 30

    if fragment_duration_s > max_fragment_duration_s:
        raise Exception(
            "Estimated fragment_duration_s is too big: %.1f (based on %s's step_duration_s=%s).\n"
            "To fix this, either:\n"
            "\t* Increase train env runners (current: %d)\n"
            "\t* Decrease train_batch_size_per_learner (current: %s)\n"
            "\t* Increase max_fragment_duration_s (current: %d, hard-coded)" % (
                fragment_duration_s, opponent, step_duration_s,
                train_env_cfg["runners"], batch_sizes,
                max_fragment_duration_s
            )
        )

    return fragment_duration_s


def build_master_config(cfg):
    train_env_cfg = deepmerge(cfg["env"]["common"], cfg["env"]["train"])
    eval_env_cfg = deepmerge(cfg["env"]["common"], cfg["env"]["eval"])
    fragment_duration_s = calculate_fragment_duration_s(cfg, train_env_cfg, eval_env_cfg)

    # Estimation is for Mac M1, ensure enough headroom for slower CPUs
    fragment_duration_headroom = 10
    sample_timeout_s = fragment_duration_s * fragment_duration_headroom
    print(f"Using sample_timeout_s={sample_timeout_s:.1f}")

    return dict(
        # Most of the `resources()` is moved to `env_runners()`
        resources=dict(
            num_cpus_for_main_process=1,
            placement_strategy="PACK",
        ),
        framework=dict(
            framework="torch",
        ),
        api_stack=dict(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        ),
        environment=dict(
            env="VCMI",
            env_config=train_env_cfg["kwargs"],
            action_mask_key="action_mask",  # not used as of ray-2.38.0
            action_space=None,              # inferred from env
            clip_actions=False,             #
            clip_rewards=None,              #
            disable_env_checking=False,     # one-time calls to step and reset
            is_atari=False,                 #
            normalize_actions=False,        # ignored for Discrete action space
            observation_space=None,         # inferred from env
            render_env=False,               #
        ),
        env_runners=dict(
            add_default_connectors_to_env_to_module_pipeline=True,
            add_default_connectors_to_module_to_env_pipeline=True,
            batch_mode="truncate_episodes",
            compress_observations=False,
            custom_resources_per_env_runner={},
            env_runner_cls=None,
            env_to_module_connector=None,
            episode_lookback_horizon=1,
            explore=False,
            max_requests_in_flight_per_env_runner=1,  # buggy, see https://github.com/ray-project/ray/pull/48499
            module_to_env_connector=None,
            num_env_runners=train_env_cfg["runners"],
            num_cpus_per_env_runner=cfg["env"]["cpu_demand"][train_env_cfg["kwargs"]["opponent"]],
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=1,  # i.e. vec_env.num_envs. MUST BE 1 (or amend fragment_size)
            rollout_fragment_length="auto",  # manually choosing a value is too complex
            update_worker_filter_stats=True,
            use_worker_filter_stats=True,
            sample_timeout_s=sample_timeout_s,
            validate_env_runners_after_construction=True,
        ),
        learners=dict(
            num_learners=0,             # 0 => learn in main process
            num_gpus_per_learner=0,
            num_cpus_per_learner=1,
        ),
        # !!! This is the API as of ray-2.38.0
        # !!! It *will* change in future releases
        training=dict(
            clip_param=cfg["clip_param"],
            entropy_coeff=cfg["entropy_coeff"],
            gamma=cfg["gamma"],
            grad_clip=cfg["grad_clip"],
            grad_clip_by=cfg["grad_clip_by"],
            kl_coeff=cfg["kl_coeff"],
            kl_target=cfg["kl_target"],
            lambda_=cfg["lambda_"],
            lr=cfg["lr"],
            minibatch_size=cfg["minibatch_size"],
            num_epochs=cfg["num_epochs"],
            train_batch_size_per_learner=cfg["train_batch_size_per_learner"],  # i.e. batch_size; or mppo's num_steps when n_envs=1
            shuffle_batch_per_epoch=cfg["shuffle_batch_per_epoch"],
            use_critic=cfg["use_critic"],
            use_gae=cfg["use_gae"],
            use_kl_loss=cfg["use_kl_loss"],
            vf_clip_param=cfg["vf_clip_param"],
            vf_loss_coeff=cfg["vf_loss_coeff"],
        ),
        multi_agent=dict(),
        offline_data=dict(),
        evaluation=dict(
            evaluation_interval=1,  # !!! MUST BE 1
            evaluation_num_env_runners=eval_env_cfg["runners"],
            evaluation_duration=cfg["evaluation_episodes"],
            evaluation_duration_unit="episodes",
            evaluation_sample_timeout_s=120.0,
            evaluation_parallel_to_training=False,
            evaluation_force_reset_envs_before_iteration=True,
            evaluation_config=MPPO_Config.overrides(
                explore=False,
                env_config=eval_env_cfg["kwargs"],
                num_cpus_per_env_runner=cfg["env"]["cpu_demand"][eval_env_cfg["kwargs"]["opponent"]],
                num_gpus_per_env_runner=0,
                num_envs_per_env_runner=1,
            ),

            # TODO: my evaluator function here?
            #       For testing siege/no siege, also logging the custom metrics
            custom_evaluation_function=None,

            # off_policy_estimation_methods={},     # relevant for offline observations
            # ope_split_batch_by_episode=True,      # relevant for offline observations
        ),
        reporting=dict(
            # metrics_num_episodes_for_smoothing=6,   # auto-set (custom logic)
            keep_per_episode_custom_metrics=False,
            log_gradients=True,

            # metrics_episode_collection_timeout_s=60.0,  # seems old API

            # Used to call .training_step() multiple times in one .train(),
            # but MPPO_Algorithm expects 1 training_step() call (loops internally)
            # (see user.training_step_duration_s)
            min_time_s_per_iteration=None,
            min_train_timesteps_per_iteration=0,
            min_sample_timesteps_per_iteration=0,
        ),
        checkpointing=dict(
            export_native_model_files=False,            # for torch .pt files in checkpoints
            checkpoint_trainable_policies_only=False,
        ),
        debugging=dict(
            # MPPO_Logger currently logs nothing.
            # It's mostly used to supressing a deprecation warning.
            logger_config=dict(type=MPPO_Logger, prefix="MPPO_Logger_prefix"),
            log_level="DEBUG",
            log_sys_usage=False,
            seed=None,
        ),
        fault_tolerance=dict(
            recreate_failed_env_runners=False,      # XXX: useful for Amazon SPOT
            ignore_env_runner_failures=False,
            max_num_env_runner_restarts=1000,
            delay_between_env_runner_restarts_s=10.0,
            restart_failed_sub_environments=False,
            num_consecutive_env_runner_failures_tolerance=100,
            env_runner_health_probe_timeout_s=30,
            env_runner_restore_timeout_s=1800,
        ),
        rl_module=dict(
            model_config=dict(
                vf_share_layers=cfg["vf_share_layers"],
                obs_dims=dict(
                    misc=env_cls.STATE_SIZE_MISC,
                    stacks=env_cls.STATE_SIZE_STACKS,
                    hexes=env_cls.STATE_SIZE_HEXES,
                ),
                network=cfg["network"]
            )
        ),
        # MPPO-specific cfg (not default to ray)
        user=dict(
            experiment_name="newray-test",
            training_step_duration_s=cfg["training_step_duration_s"],
            wandb_project=None,
            wandb_old_run_id=None,
            wandb_log_interval_s=cfg["wandb_log_interval_s"],
            hyperparam_mutations=cfg["hyperparam_mutations"]
        )
    )


def convert_to_param_space(mutations):
    res = {}
    for k, v in mutations.items():
        if isinstance(v, dict):
            res[k] = convert_to_param_space(v)
        elif isinstance(v, list):
            res[k] = ray.tune.choice(v)
        else:
            assert isinstance(v, ray.tune.search.sample.Domain)
    return res


# Even if there are remote runners, ray still creates 2 local runners
# It does not use them for anything except inferring action/observation spaces
class DummyEnv(env_cls):
    def __init__(self, *args, **kwargs):
        self.action_space = env_cls.ACTION_SPACE
        self.observation_space = env_cls.OBSERVATION_SPACE
        pass

    def step(self, *args, **kwargs):
        raise Exception("step() called on DummyEnv")

    def reset(self, *args, **kwargs):
        raise Exception("reset() called on DummyEnv")

    def render(self, *args, **kwargs):
        raise Exception("render() called on DummyEnv")

    def close(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    algo_config = MPPO_Config()
    algo_config.callbacks(MPPO_Callback)  # this cannot go in master_config
    algo_config.master_config(build_master_config(pbt_config))

    # ray.tune.registry.register_env("VCMI", lambda cfg: (print("NEW ENV WITH INDEX: %s" % cfg.worker_index), env_cls(**cfg)))
    def env_creator(cfg):
        if cfg.num_workers > 0 and cfg.worker_index == 0:
            return DummyEnv()
        else:
            return env_cls(**cfg)

    ray.tune.registry.register_env("VCMI", env_creator)
    ray.tune.registry.register_trainable("MPPO", MPPO_Algorithm)

    checkpoint_config = ray.train.CheckpointConfig(num_to_keep=3)
    run_config = ray.train.RunConfig(
        name="newray-test",
        verbose=False,
        failure_config=ray.train.FailureConfig(max_failures=-1),
        checkpoint_config=checkpoint_config,
        # stop={"rew_mean": pbt_config["_raytune"]["target_ep_rew_mean"]},
        # callbacks=[TBXDummyCallback()],
        storage_path=os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    )

    # Hyperparams are used twice here:
    # 1. In PBT's `hyperparam_mutations` (as lists of values) - used
    #    for perturbations.
    # 2. In Tuner's `param_space` (as tune.choice objects) - used
    #    for initial sampling

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    scheduler = ray.tune.schedulers.PopulationBasedTraining(
        metric=pbt_config["metric"],
        mode="max",
        time_attr="training_iteration",
        perturbation_interval=1,
        hyperparam_mutations=algo_config.user["hyperparam_mutations"],
        quantile_fraction=pbt_config["quantile_fraction"],
        log_config=True,  # used for reconstructing the cfg schedule
        require_attrs=True,
        synch=True,
    )

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html#ray.tune.TuneConfig
    tune_config = ray.tune.TuneConfig(
        trial_name_creator=lambda t: t.trial_id,
        trial_dirname_creator=lambda t: t.trial_id,
        scheduler=scheduler,
        reuse_actors=False,  # XXX: False is much safer and ensures no state leaks
        num_samples=pbt_config["population_size"],
        search_alg=None
    )

    #
    # XXX: no use in limiting cluster or worker resources
    #      Calculating the appropriate population_size is enough
    #      (it is essentially the limit for number of spawned workers)
    #      => don't impose any additional limits here to avoid confusion
    #      GPU must be non-0 to be available at all => set to 0.01 if cuda is available
    #
    # ray.init() by default uses num_cpus=os.cpu_count(), num_gpus=torch.cuda.device_count()
    # However, if GPU is 0 then CUDA is always unavailable in the workers => set to 0.01
    #
    # resources = ray.tune.PlacementGroupFactory([{
    #     "CPU": 0.01,
    #     "GPU": 0.01 if torch.cuda.is_available() else 0
    # }])

    # trainable = ray.tune.with_parameters(trainable_cls, initargs=initargs)
    # trainable = ray.tune.with_resources(trainable, resources=resources)

    for k, v in convert_to_param_space(algo_config.user["hyperparam_mutations"]).items():
        assert hasattr(algo_config, k)
        if isinstance(v, dict):
            deep_update(getattr(algo_config, k), v)
        else:
            setattr(algo_config, k, v)

    tuner = ray.tune.Tuner(
        trainable="MPPO",
        run_config=run_config,
        tune_config=tune_config,
        param_space=algo_config,
    )

    tuner.fit()
