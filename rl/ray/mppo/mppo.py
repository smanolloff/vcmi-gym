import re
import wandb
from dataclasses import dataclass
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

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
        wandb_project: str | None,
        wandb_old_run_id: str | None,
        wandb_log_interval: int = 50,  # in train iterations (i.e. rollouts)
    ):
        self.user = {
            "experiment_name": experiment_name,
            "wandb_project": wandb_project,
            "wandb_old_run_id": wandb_old_run_id,
            "wandb_log_interval": wandb_log_interval,
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
    def master_config(self, **cfg):
        assert hasattr(self, "_master_config") is False, "master_config() must be called exactly once"
        self._master_config = cfg

        for k, v in cfg.items():
            getattr(self, k)(**v)

        # Make sure all evaluation results will fit into the metric window
        # (discarding them is a pure waste of computing power)
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
        assert self.user["wandb_log_interval"] > 0
        assert re.match(r"^[\w_-]+$", self.user["experiment_name"]), self.user["experiment_name"]
        if self.user["wandb_project"] is not None:
            assert re.match(r"^[\w_-]+$", self.user["wandb_project"]), self.user["wandb_project"]
        if self.user["wandb_old_run_id"] is not None:
            assert re.match(r"^[\w_-]+$", self.user["wandb_old_run_id"]), self.user["wandb_old_run_id"]


class MPPO_Algorithm(PPO):
    # Variable storage to prevent collisions with variables upstream
    @dataclass
    class Namespace:
        master_config: dict
        run_id: str
        run_name: str
        log_interval: int

    @classmethod
    @override(PPO)
    def get_default_config(cls):
        return MPPO_Config()

    @override(PPO)
    def setup(self, config: MPPO_Config):
        assert hasattr(config, "_master_config"), "call .master_config() on the MPPO_Config first"
        self._wandb_init(config)
        super().setup(config)  # *after* wandb init

    #
    # private
    #

    def _wandb_init(self, config):
        old_run_id = config.user["wandb_old_run_id"]

        if old_run_id:
            assert re.match(r"^[0-9a-z]+_[0-9]+$", old_run_id), f"bad id to resume: {old_run_id}"
            run_id = "%s_%s" % (old_run_id.split("_")[0], self.trial_id.split("_")[1])
            print("Will resume run as id %s (Trial ID is %s)" % (run_id, self.trial_id))
        else:
            run_id = self.trial_id
            print("Will start new run %s" % run_id)

        run_name = self.trial_name
        if self.trial_name != "default":
            run_name = "T%d" % int(self.trial_name.split("_")[-1])

        self.ns = MPPO_Algorithm.Namespace(
            master_config=config._master_config,
            run_id=run_id,
            run_name=run_name,
            log_interval=config.user["wandb_log_interval"]
        )

        if config.user["wandb_project"]:
            wandb.init(
                project=config.user["wandb_project"],
                group=config.user["experiment_name"],
                id=self.ns.run_id,
                name=self.ns.run_name,
                resume="allow",
                reinit=True,
                allow_val_change=True,
                settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                config=self.ns.master_config,
                sync_tensorboard=False,
            )
