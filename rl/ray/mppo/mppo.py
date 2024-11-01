import re
import time
from dataclasses import dataclass
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from wandb.util import json_dumps_safer

from .mppo_rl_module import MPPO_RLModule
# from .mppo_learner import MPPO_Learner


class MPPO_Config(PPOConfig):
    def __init__(self):
        super().__init__(algo_class=MPPO_Algorithm)
        self.enable_rl_module_and_learner = True
        self.enable_env_runner_and_connector_v2 = True

    @override(PPOConfig)
    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=MPPO_RLModule)

    @property
    @override(PPOConfig)
    def _model_config_auto_includes(self):
        return {}

    # User-defined config (not used by ray)
    def user(
        self,
        experiment_name: str,
        training_step_duration_s: int,
        hyperparam_mutations: dict,
        wandb_project: str | None,
        wandb_old_run_id: str | None,
        wandb_log_interval_s: int,  # in train iterations (i.e. rollouts)
    ):
        self.user = {
            "experiment_name": experiment_name,
            "training_step_duration_s": training_step_duration_s,
            "hyperparam_mutations": hyperparam_mutations,
            "wandb_project": wandb_project,
            "wandb_old_run_id": wandb_old_run_id,
            "wandb_log_interval_s": wandb_log_interval_s,
        }

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
        assert hasattr(self, "_master_config") is False, "master_config() must be called exactly once"
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

    #
    # private
    #

    def _validate(self):
        assert self.evaluation_interval == 1, "Tune expects eval results on each iteration"
        assert self.user["training_step_duration_s"] > 0
        assert self.user["wandb_log_interval_s"] > 0
        assert self.user["wandb_log_interval_s"] < self.user["training_step_duration_s"]
        assert re.match(r"^[\w_-]+$", self.user["experiment_name"]), self.user["experiment_name"]
        if self.user["wandb_project"] is not None:
            assert re.match(r"^[\w_-]+$", self.user["wandb_project"]), self.user["wandb_project"]
        if self.user["wandb_old_run_id"] is not None:
            assert re.match(r"^[\w_-]+$", self.user["wandb_old_run_id"]), self.user["wandb_old_run_id"]
        if self.num_learners > 0:
            # We can't setup wandb via self.learner_group.foreach_learner(...)
            #   1. if worker is restarted, it won't have wandb setup
            #       (can't use Callbacks.on_workers_recreated as it's for EnvRunners)
            #   2. wandb login must be ensured on all remotes prior to ray start
            #   3. I don't understand how multi-learner setup works yet
            #       (e.g. how are losses/gradients from N learners combined)
            raise Exception("TODO(simo): wandb setup in remote learners is not implemented")


class MPPO_Algorithm(PPO):
    # Variable storage to prevent collisions with variables upstream
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
    def setup(self, config: MPPO_Config):
        assert hasattr(config, "_master_config"), "call .master_config() on the MPPO_Config first"
        self.ns = MPPO_Algorithm.Namespace(
            master_config=config._master_config,
            log_interval=config.user["wandb_log_interval_s"]
        )
        print("*** SETUP: %s" % json_dumps_safer(config.to_dict()))
        super().setup(config)
        self.wandb_log = self.callbacks.wandb_log
        print("*********** VF_CLIP_PARAM IS: %s" % self.config.vf_clip_param)

        # TODO: implement restore of previously trained model + optimizer
        #       (for a NEW experiment, not resuming):
        #           self.restore_from_path("data/newray-test/01800_00001/checkpoint_000000")
        #
        #       Optionally, reset iteration:
        #           self._iteration == 0
        #
        #       Optionally, load learners only:
        #           path = "data/newray-test/01800_00001/checkpoint_000000/learner_group"
        #           self.learner_group.restore_from_path()

    @override(PPO)
    def training_step(self):
        # XXX: there's no `on_training_step_start` callback => log this here
        self.wandb_log({"trial/iteration": self.iteration})

        temp_logger = MetricsLogger()
        started_at = time.time()
        logged_at = started_at
        training_step_duration_s = self.config.user["training_step_duration_s"]
        wandb_log_interval_s = self.config.user["wandb_log_interval_s"]

        # XXX: this will be problematic with ray SPOT instances where
        #      different runners may run with different speeds?
        while True:
            temp_logger.log_dict(super().training_step())
            now = time.time()

            # Call custom MPPO-specific callback once every N seconds
            if (now - logged_at) > wandb_log_interval_s:
                self.callbacks.on_train_subresult(self, temp_logger.reduce(return_stats_obj=False))
                logged_at = now

            if (now - started_at) > training_step_duration_s:
                break

        return temp_logger.reduce()

    @override(PPO)
    def save_checkpoint(self, checkpoint_dir):
        import ipdb; ipdb.set_trace()  # noqa
        super().save_checkpoint(checkpoint_dir)
