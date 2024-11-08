import datetime
import os
import pygit2
import re
import tempfile
import threading
import time
import wandb

from dataclasses import dataclass, asdict
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.policy.sample_batch import (
    DEFAULT_POLICY_ID
)

from ray.tune.result import TRIAL_INFO
from wandb.util import json_dumps_safer

from .mppo_rl_module import MPPO_RLModule
from .mppo_env_runners import MPPO_TrainEnv, MPPO_EvalEnv
from .mppo_logger import MPPO_Logger, get_logger
from .mppo_callback import MPPO_Callback
from . import util


class MPPO_Config(PPOConfig):
    @dataclass
    class UserConfig:
        # General
        env_runner_keepalive_interval_s: int
        experiment_name: str
        wandb_project: str | None
        wandb_log_interval_s: int

        # Tune
        hyperparam_mutations: dict
        hyperparam_values: dict
        population_size: int
        quantile_fraction: float
        training_step_duration_s: int

        # Other
        git_head: str
        git_is_dirty: bool
        master_overrides: dict
        model_load_file: str | None
        checkpoint_load_dir: str | None
        init_argument: str
        init_method: str
        timestamp: str

        # XXX: validations support primitive data types only
        def __post_init__(self):
            util.validate_dataclass_fields(self)

        def json_encode(self):
            return asdict(self)

    def __init__(self):
        super().__init__(algo_class=MPPO_Algorithm)
        self.enable_rl_module_and_learner = True
        self.enable_env_runner_and_connector_v2 = True
        self._master_config = None

        env_id = "VCMI-v4"
        env_cls = util.get_env_cls(env_id)
        env_cfg = {"conntype": "proc"}

        (
            # Default config
            self
            .resources(
                num_cpus_for_main_process=1,
                placement_strategy="PACK")
            .framework(
                framework="torch")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True)
            .environment(
                env=env_id,
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
                env_runner_cls=MPPO_TrainEnv,
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
            # !!! This is the API as of ray-2.38.0
            # !!! It *will* change in future releases
            .training(
                clip_param=0.3,
                entropy_coeff=0.0,
                gamma=0.8,
                grad_clip=5,
                grad_clip_by="global_norm",  # global_norm = nn.utils.clip_grad_norm_(model.parameters)
                kl_coeff=0.2,
                kl_target=0.01,
                lambda_=1.0,
                lr=0.001,
                minibatch_size=20,
                num_epochs=1,
                train_batch_size_per_learner=500,
                shuffle_batch_per_epoch=True,
                use_critic=True,
                use_gae=True,
                use_kl_loss=True,
                vf_clip_param=10.0,
                vf_loss_coeff=1.0)
            .callbacks(MPPO_Callback)
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
                    env_runner_cls=MPPO_EvalEnv),
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
                # MPPO_Logger currently logs nothing.
                # It's mostly used to supressing a deprecation warning.
                logger_config=dict(type=MPPO_Logger, prefix="MPPO_Logger_prefix"),
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
                env_runner_keepalive_interval_s=15,
                experiment_name="MPPO-default",
                wandb_project=None,
                wandb_log_interval_s=60,

                hyperparam_mutations={},
                hyperparam_values={},
                population_size=1,
                quantile_fraction=0.5,
                training_step_duration_s=3600,

                git_head="",
                git_is_dirty=False,
                master_overrides={},
                model_load_file=None,
                checkpoint_load_dir=None,
                init_argument="default",
                init_method="default",
                timestamp="2000-01-01T00:00:00")
        )

    @override(PPOConfig)
    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=MPPO_RLModule)

    @property
    @override(PPOConfig)
    def _model_config_auto_includes(self):
        return {}

    # User-defined config (not used by ray)
    # Keys are passed as regular key-value pairs for a reason:
    # they are part of "master_config" and must be JSON-serializable.
    # The `User` data structure is used for validation purposes.
    def user(self, **kwargs):
        # History items may not conform to the current InitInfo structure
        # => leave them as plain dicts
        self.user_config = self.UserConfig(**kwargs)
        return self

    #
    # Usage:
    # mppo_config = MPPO_Config()
    # mppo_config.master_config(
    #   "resources": {...},         # instead of mppo_config.resources(...)
    #   "environment": {...},       # instead of mppo_config.environment(...)
    #   ...                         # ...etc
    # )
    #
    def master_config(self, cfg):
        self._master_config = cfg

        for k, v in cfg.items():
            getattr(self, k)(**v)

        # Make sure all evaluated episodes fit into the metric window
        # (discarding them is a pure waste of resources)
        if isinstance(self.evaluation_duration, int) and self.evaluation_duration_unit == "episodes":
            self.metrics_num_episodes_for_smoothing = self.evaluation_duration
            # assert self.metrics_num_episodes_for_smoothing >= self.evaluation_duration, (
            #     f"{self.metrics_num_episodes_for_smoothing} >= {self.evaluation_duration}"
            # )

        self._validate()

    # XXX: Temp fix until https://github.com/ray-project/ray/pull/48529 is merged
    def update_from_dict(self, config):
        if TRIAL_INFO in config:
            setattr(self, TRIAL_INFO, config[TRIAL_INFO])
        return super().update_from_dict(config)

    #
    # private
    #

    def _validate(self):
        assert self.evaluation_interval == 1, "Tune expects eval results on each iteration"

        uc = self.user_config

        assert uc.training_step_duration_s >= 0
        assert uc.wandb_log_interval_s >= 0
        assert uc.wandb_log_interval_s <= uc.training_step_duration_s
        assert re.match(r"^[\w_-]+$", uc.experiment_name), uc.experiment_name

        if uc.wandb_project is not None:
            assert re.match(r"^[\w_-]+$", uc.wandb_project), uc.wandb_project

        if self.num_learners > 0:
            # We can't setup wandb via self.learner_group.foreach_learner(...)
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

        validate_hyperparam_mutations(self.user_config.hyperparam_mutations or {})

        if self.env_config["conntype"] == "thread" and self.num_env_runners == 0:
            raise Exception("Train runners are local and cannot have conntype='thread'")

        if self.evaluation_config["env_config"]["conntype"] == "thread" and self.evaluation_num_env_runners == 0:
            raise Exception("Eval runners are local and cannot have conntype='thread'")


class MPPO_Algorithm(PPO):
    @dataclass
    class Namespace:
        master_config: dict
        log_interval: int
        run_id: str = None
        run_name: str = None

    @classmethod
    @override(PPO)
    def get_default_config(cls):
        return MPPO_Config()

    @override(PPO)
    def __init__(self, *args, **kwargs):
        util.silence_log_noise()
        super().__init__(*args, **kwargs)

    @override(PPO)
    def setup(self, config: MPPO_Config):
        print("SELF.TRIAL_ID: %s" % self.trial_id)
        self.ns = MPPO_Algorithm.Namespace(
            master_config=config._master_config,
            log_interval=config.user_config.wandb_log_interval_s
        )

        # self.logger.debug("*** SETUP: %s" % json_dumps_safer(config.to_dict()))
        super().setup(config)
        self._wandb_init()
        self.logger = get_logger(self.ns.run_id, "DEBUG")  # *after* wandb init
        self.logger.info("Logger initialized")

        # Must be *after* super().setup()
        self._wandb_add_watch()
        self._wandb_log_hyperparams()

    @override(PPO)
    def training_step(self):
        # XXX: there's no `on_training_step_start` callback => log this here
        self.wandb_log({"trial/iteration": self.iteration})

        if os.getenv("DEBUG"):
            if self.iteration == 0:
                # XXX: self.iteration is always 0 during setup(), must load here
                self._maybe_load_learner_group()
                self._maybe_load_model()

        started_at = time.time()
        logged_at = started_at
        training_step_duration_s = self.config.user_config.training_step_duration_s
        wandb_log_interval_s = self.config.user_config.wandb_log_interval_s
        keepalive_interval_s = self.config.user_config.env_runner_keepalive_interval_s

        with EnvRunnerKeepalive(self.eval_env_runner_group, keepalive_interval_s, self.logger):
            # XXX: will this fixed-time step be problematic with ray SPOT
            #       instances where different runners may run with different
            #       speeds?
            while True:
                time.sleep(5)
                result = super().training_step()
                now = time.time()

                # Call custom MPPO-specific callback once every N seconds
                if (now - logged_at) > wandb_log_interval_s:
                    # NOTE: do not call self.metrics.reduce (resets values)
                    self.callbacks.on_train_subresult(self, result)
                    logged_at = now

                self.logger.debug("training_step time left: %ds" % (training_step_duration_s - (now - started_at)))
                if (now - started_at) > training_step_duration_s:
                    break

        return result

    @override(PPO)
    def evaluate(self, *args, **kwargs):
        keepalive_interval_s = self.config.user_config.env_runner_keepalive_interval_s
        with EnvRunnerKeepalive(self.env_runner_group, keepalive_interval_s, self.logger):
            return super().evaluate(*args, **kwargs)

    @override(PPO)
    def save_checkpoint(self, checkpoint_dir):
        res = super().save_checkpoint(checkpoint_dir)

        if not (wandb.run and re.match(r"^.+_0+$", self.trial_id)):
            return res

        learner = self.learner_group.get_checkpointable_components()[0][1]

        rl_module = learner.module[DEFAULT_MODULE_ID]
        model_file = os.path.join(checkpoint_dir, "jit-model.pt")
        rl_module.jsave(model_file)

        config_file = os.path.join(checkpoint_dir, "master_config.json")

        # TRIAL_INFO key contains non-serializable (by wandb) values
        with open(config_file, "w") as f:
            f.write(json_dumps_safer({k: v for k, v in self.config.to_dict().items() if k != TRIAL_INFO}))

        art = wandb.Artifact(name="model", type="model")
        art.description = f"Snapshot of model from {time.ctime(time.time())}"
        art.ttl = datetime.timedelta(days=7)
        art.metadata["step"] = wandb.run.step
        art.add_dir(checkpoint_dir)
        wandb.run.log_artifact(art)

        return res

    # XXX: in case of SIGTERM/SIGINT, ray does not wait for cleanup to finish.
    #      During regular perturbation, though, it does.
    @override(PPO)
    def cleanup(self, *args, **kwargs):
        if wandb.run:
            wandb.finish(quiet=True)
        return super().cleanup(*args, **kwargs)

    #
    # private
    #

    def _wandb_init(self):
        run_id = util.gen_id() if self.trial_id == "default" else self.trial_id
        print("W&B Run ID is %s" % run_id)

        run_name = self.trial_name
        if self.trial_name == "default":
            run_name = f"{datetime.datetime.now().isoformat()}-debug-{run_id}"
        else:
            run_name = "T%d" % int(self.trial_name.split("_")[-1])

        self.ns.run_id = run_id
        self.ns.run_name = run_name

        if self.config.user_config.wandb_project:
            wandb.init(
                project=self.config.user_config.wandb_project,
                group=self.config.user_config.experiment_name,
                id=gen_id() if self.ns.run_id == "default" else self.ns.run_id,
                name=self.ns.run_name,
                resume="allow",
                reinit=True,
                allow_val_change=True,
                settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                config=self.ns.master_config,
                sync_tensorboard=False,
            )

            self._wandb_log_git_diff()  # superseded by "code saving" profile setting

            # For wandb.log, commit=True by default
            # for wandb_log, commit=False by default
            def wandb_log(*args, **kwargs):
                wandb.log(*args, **dict({"commit": False}, **kwargs))
        else:
            def wandb_log(*args, **kwargs):
                print("*** WANDB LOG AT %s: %s %s" % (datetime.datetime.now().isoformat(), args, kwargs))

        self.wandb_log = wandb_log

    def _wandb_add_watch(self):
        if not wandb.run:
            return

        # XXX: wandb.watch() caused issues during serialization in oldray scripts
        #      (it pollutes the model with non-serializable callbacks)
        #      It seems ray's checkpointing works OK with it
        assert self.learner_group.is_local
        self.learner_group.foreach_learner(lambda l: wandb.watch(
            l.module[DEFAULT_POLICY_ID].encoder.encoder,
            log="all",
            log_graph=True,
            log_freq=1000
        ))

    def _wandb_log_hyperparams(self):
        to_log = {}

        for k, v in self.config.user_config.hyperparam_mutations.items():
            if k == "env_config":
                for k1, v1 in v.items():
                    assert k1 in self.config.env_config, f"{k1} in self.config.env_config"
                    assert "/" not in k1
                    to_log[f"params/env.{k1}"] = self.config.env_config[k1]
            else:
                assert hasattr(self.config, k), f"hasattr(self.config, {k})"
                assert "/" not in k
                to_log[f"params/{k}"] = getattr(self.config, k)

        if to_log:
            self.wandb_log(to_log)

    # XXX: There's already a wandb "code saving" profile setting
    #      It saves only requirements.txt and the git metadata which is enough
    #      See https://docs.wandb.ai/guides/track/log/
    # def _wandb_log_code(self):
    #     # https://docs.wandb.ai/ref/python/run#log_code
    #     # XXX: "path" is relative to `ray_root`
    #     this_file = pathlib.Path(__file__)
    #     ray_root = this_file.parent.parent.absolute()
    #     # TODO: log requirements.txt as well
    #     wandb.run.log_code(
    #         root=ray_root,
    #         include_fn=lambda path: path.endswith(".py"),
    #     )

    # XXX: There's already a wandb "code saving" profile setting
    #      See https://docs.wandb.ai/guides/track/log/
    #      but it's NOT WORKING (says couldnt find git commit for this run)
    #      (maybe it requires logging my entire codebase?)
    def _wandb_log_git_diff(self):
        import os
        if not os.getenv("DEBUG"):
            return

        # if self.iteration > 0 or not self._is_golden_trial():
        #     return

        git = pygit2.Repository(os.path.dirname(__file__))
        head = str(git.head.target)
        assert head == self.config.user_config.git_head

        art = wandb.Artifact(name="git-diff", type="text")
        art.description = f"Git diff for HEAD@{head} from {time.ctime(time.time())}"

        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
            art.metadata["head"] = head
            temp_file.write(git.diff().patch)
            temp_file.flush()
            art.add_file(temp_file.name, name="diff.patch")
            wandb.run.log_artifact(art)
            # no need to wait (wandb creates a local copy of the file to upload)
            # art.wait()

    # The first trial in an experiment (e.g. "ff859_00000")
    def _is_golden_trial(self):
        return re.match(r"^.+_0+$", self.trial_id)

    def _maybe_load_learner_group(self):
        return
        if "experiment" not in self.config.user["load_options"]:
            return

        path = util.to_abspath(self.config.user["load_options"]["learner_group"]["path"])
        self.logger.warning(f"Loading learner group from {path}")
        self.learner_group.restore_from_path(path)
        self._broadcast_weights()

    # XXX: this works for PPO, but do other algos use more policies?
    def _maybe_load_model(self):
        return
        if "model" not in self.config.user["load_options"]:
            return

        mapping = self.config.user["load_options"]["model"]["layer_mapping"]
        path = util.to_abspath(self.config.user["load_options"]["model"]["path"])
        self.logger.warning(f"Loading learner model from {path}")
        self.learner_group.foreach_learner(lambda l: l.module[DEFAULT_POLICY_ID].jload(path, mapping))
        self._broadcast_weights()

    def _broadcast_weights(self):
        opts = dict(from_worker_or_learner_group=self.learner_group, inference_only=True)

        self.logger.info("Broadcasting learner weights to env runners")
        self.env_runner_group.sync_weights(**opts)

        if self.eval_env_runner_group is not None:
            self.logger.info("Broadcasting learner weights to eval env runners")
            self.eval_env_runner_group.sync_weights(**opts)


class EnvRunnerKeepalive:
    def __init__(self, runner_group, interval, logger):
        self.runner_group = runner_group
        self.interval = interval
        self.logger = logger

    def __enter__(self):
        self.logger.info("Starting env runner keepalive loop")
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        # self.thread = threading.Thread(target=self._debug_loop, daemon=True)
        self.thread.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.info("Stopping env runner keepalive loop...")
        self.event.set()
        self.thread.join()
        self.logger.debug("Stopped env runner keepalive loop")

    def _loop(self):
        while not self.event.is_set():
            self.event.wait(timeout=self.interval)
            self.runner_group.foreach_worker(
                func=lambda w: w.ping(),
                timeout_seconds=10,  # should be instant
                local_env_runner=False,
            )

    def _debug_loop(self):
        def debug(w):
            import ipdb; ipdb.set_trace()  # noqa
        self.runner_group.foreach_worker(func=debug)
