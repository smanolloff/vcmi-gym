import wandb
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.dreamerv3 import DreamerV3, DreamerV3Config
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.result import TRIAL_INFO
from ray.rllib.core import DEFAULT_MODULE_ID

from .mdreamerv3_callback import MDreamerV3_Callback
from .mdreamerv3_rl_module import MDreamerV3_RLModule
from .mdreamerv3_env_runner import MDreamerV3_EnvRunner
from .mdreamerv3_episode_replay_buffer import MDreamerV3_EpisodeReplayBuffer

from ..common import common_config, common_algorithm, common_logger, util


class MDreamerV3_Config(DreamerV3Config):
    @override(DreamerV3Config)
    def __init__(self):
        super().__init__(algo_class=MDreamerV3_Algorithm)
        common_config.init(self, MDreamerV3_Callback)

        # !!! This is the API as of ray-2.38.0
        # !!! It *will* change in future releases
        (
            # Default config
            self
            .training(
                model_size="XS",
                training_ratio=32,
                gc_frequency_train_steps=100,
                batch_size_B=16,
                batch_length_T=64,
                horizon_H=15,
                gae_lambda=0.95,
                entropy_scale=3e-4,
                return_normalization_decay=0.99,
                train_critic=True,
                train_actor=True,
                intrinsic_rewards_scale=0.1,
                world_model_lr=1e-4,
                actor_lr=3e-5,
                critic_lr=3e-5,
                world_model_grad_clip_by_global_norm=1000.0,
                critic_grad_clip_by_global_norm=100.0,
                actor_grad_clip_by_global_norm=100.0,
                symlog_obs=True,
                use_float16=False,

                # XXX: in VCMI, 1 obs ~= 40KB
                # => 1M obs = 40GB...
                # => default to 10K (4GB)
                # TODO: try use_float16=True to see if it halves the obs size
                replay_buffer_config={
                    "type": "EpisodeReplayBuffer",
                    "capacity": 100_000,
                })
            .framework(framework="tf2")
            .env_runners(
                num_env_runners=0,  # errors if >0
                env_runner_cls=MDreamerV3_EnvRunner,
                rollout_fragment_length=100)
            .evaluation(
                evaluation_num_env_runners=0,  # errors if >0
                evaluation_interval=2,  # !!! MUST BE 1
                evaluation_config=dict(env_runner_cls=MDreamerV3_EnvRunner)
            )
            .rl_module(rl_module_spec=RLModuleSpec(module_class=MDreamerV3_RLModule))
        )

    @override(DreamerV3Config)
    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=MDreamerV3_RLModule)

    # @property
    # @override(DreamerV3Config)
    # def _model_config_auto_includes(self):
    #     return {}

    def user(self, **kwargs):
        return common_config.configure_user(self, kwargs)

    def master_config(self, cfg):
        return common_config.configure_master(self, cfg)

    def update_from_dict(self, config):
        if TRIAL_INFO in config:
            setattr(self, TRIAL_INFO, config[TRIAL_INFO])
        return super().update_from_dict(config)

    def _validate(self):
        return common_config.validate(self)


class MDreamerV3_Algorithm(DreamerV3):
    ALGO_NAME = "MDreamerV3"

    @classmethod
    @override(DreamerV3)
    def get_default_config(cls):
        return MDreamerV3_Config()

    @override(DreamerV3)
    def __init__(self, *args, **kwargs):
        util.silence_log_noise()
        super().__init__(*args, **kwargs)

    @override(DreamerV3)
    def setup(self, config: MDreamerV3_Config):
        print("trial_id: %s" % self.trial_id)
        self.ns = common_algorithm.Namespace(
            master_config=config._master_config,
            log_interval=config.user_config.wandb_log_interval_s
        )

        # algo.logger.debug("*** SETUP: %s" % wandb.util.json_dumps_safer(config.to_dict()))
        super().setup(config)

        self.replay_buffer = MDreamerV3_EpisodeReplayBuffer(
            capacity=self.config.replay_buffer_config["capacity"],
            batch_size_B=self.config.batch_size_B,
            batch_length_T=self.config.batch_length_T,
        )

        # always true (false causes errors)
        if self.config.share_module_between_env_runner_and_learner:
            assert self.eval_env_runner.module is None
            self.eval_env_runner.module = self.learner_group._learner.module[DEFAULT_MODULE_ID]

        common_algorithm.wandb_init(self)
        self.logger = common_logger.get_logger(self.ns.run_id, "DEBUG")  # *after* wandb init
        self.logger.info("Logger initialized")

        # Must be *after* super().setup()
        # common_algorithm.wandb_add_watch(self)
        common_algorithm.wandb_log_hyperparams(self)

    @override(DreamerV3)
    def training_step(self):
        # XXX: there's no `on_training_step_start` callback => log this here
        self.wandb_log({"trial/iteration": self.iteration})

        if self.iteration == 0:
            # XXX: self.iteration is always 0 during setup(), must load here
            common_algorithm.maybe_load_learner_group(self)
            if self.config.user_config.model_load_file:
                # Save/load models is not implemented yet
                # (does not affect loading checkpoints and resuming experiments)
                raise NotImplementedError("MDreamerV3 does not support model save/load (yet)")

            # Just keep it running forever
            self._env_runner_keepalive = common_algorithm.EnvRunnerKeepalive(
                self.eval_env_runner_group,
                self.config.user_config.env_runner_keepalive_interval_s,
                self.logger
            )

        result = super().training_step()
        return result

    @override(DreamerV3)
    def evaluate(self, *args, **kwargs):
        keepalive_interval_s = self.config.user_config.env_runner_keepalive_interval_s
        with common_algorithm.EnvRunnerKeepalive(self.env_runner_group, keepalive_interval_s, self.logger):
            return super().evaluate(*args, **kwargs)

    @override(DreamerV3)
    def save_checkpoint(self, checkpoint_dir):
        res = super().save_checkpoint(checkpoint_dir)
        # dreamer is TF + it has several NNs, very different from JitModel
        common_algorithm.save_checkpoint(self, checkpoint_dir, jsave=False)
        return res

    # XXX: in case of SIGTERM/SIGINT, ray does not wait for cleanup to finish.
    #      During regular perturbation, though, it does.
    @override(DreamerV3)
    def cleanup(self, *args, **kwargs):
        if wandb.run:
            wandb.finish(quiet=True)
        return super().cleanup(*args, **kwargs)

    @override(DreamerV3)
    def __setstate__(self, state):
        super().__setstate__(state)
        if self.config.share_module_between_env_runner_and_learner:
            assert id(self.eval_env_runner.module) != id(
                self.learner_group._learner.module[DEFAULT_MODULE_ID]
            )
            self.eval_env_runner.module = self.learner_group._learner.module[DEFAULT_MODULE_ID]
