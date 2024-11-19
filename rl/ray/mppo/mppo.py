import wandb
import time
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.result import TRIAL_INFO

from .mppo_callback import MPPO_Callback
from ..common import common_config, common_algorithm, common_logger, util
from .mppo_rl_module import MPPO_RLModule


class MPPO_Config(PPOConfig):
    @override(PPOConfig)
    def __init__(self):
        super().__init__(algo_class=MPPO_Algorithm)
        common_config.init(self, MPPO_Callback)

        # !!! This is the API as of ray-2.38.0
        # !!! It *will* change in future releases
        self.training(
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
            vf_loss_coeff=1.0
        )

    @override(PPOConfig)
    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=MPPO_RLModule)

    @property
    @override(PPOConfig)
    def _model_config_auto_includes(self):
        return {}

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


class MPPO_Algorithm(PPO):
    ALGO_NAME = "MPPO"

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
        print("trial_id: %s" % self.trial_id)
        self.ns = common_algorithm.Namespace(
            master_config=config._master_config,
            log_interval=config.user_config.wandb_log_interval_s
        )

        # algo.logger.debug("*** SETUP: %s" % wandb.util.json_dumps_safer(config.to_dict()))
        super().setup(config)

        common_algorithm.wandb_init(self)
        self.logger = common_logger.get_logger(self.ns.run_id, "DEBUG")  # *after* wandb init
        self.logger.info("Logger initialized")

        # Must be *after* super().setup()
        common_algorithm.wandb_add_watch(self)
        common_algorithm.wandb_log_hyperparams(self)

    @override(PPO)
    def training_step(self):
        # XXX: there's no `on_training_step_start` callback => log this here
        self.wandb_log({"trial/iteration": self.iteration})
        print(f"Training iteration: {self.iteration}")
        # import ipdb, os; ipdb.set_trace() if os.getenv("DEBUG") else ...  # noqa

        if self.iteration == 0:
            # XXX: self.iteration is always 0 during setup(), must load here
            common_algorithm.maybe_load_learner_group(self)
            common_algorithm.maybe_load_model(self)

        started_at = time.time()
        logged_at = started_at
        training_step_duration_s = self.config.user_config.training_step_duration_s
        wandb_log_interval_s = self.config.user_config.wandb_log_interval_s
        keepalive_interval_s = self.config.user_config.env_runner_keepalive_interval_s

        with common_algorithm.EnvRunnerKeepalive(self.eval_env_runner_group, keepalive_interval_s, self.logger):
            # XXX: will this fixed-time step be problematic with ray SPOT
            #       instances where different runners may run with different
            #       speeds?
            while True:
                result = super().training_step()
                now = time.time()

                # Call custom MPPO-specific callback once every N seconds
                if (now - logged_at) > wandb_log_interval_s:
                    # NOTE: do not call self.metrics.reduce (resets values)
                    self.callbacks.on_train_subresult(self, result)
                    logged_at = now

                # self.logger.debug("training_step time left: %ds" % (training_step_duration_s - (now - started_at)))
                if (now - started_at) > training_step_duration_s:
                    break

        return result

    @override(PPO)
    def evaluate(self, *args, **kwargs):
        keepalive_interval_s = self.config.user_config.env_runner_keepalive_interval_s
        with common_algorithm.EnvRunnerKeepalive(self.env_runner_group, keepalive_interval_s, self.logger):
            return super().evaluate(*args, **kwargs)

    @override(PPO)
    def save_checkpoint(self, checkpoint_dir):
        res = super().save_checkpoint(checkpoint_dir)
        common_algorithm.save_checkpoint(self, checkpoint_dir)
        return res

    # XXX: in case of SIGTERM/SIGINT, ray does not wait for cleanup to finish.
    #      During regular perturbation, though, it does.
    @override(PPO)
    def cleanup(self, *args, **kwargs):
        if wandb.run:
            wandb.finish(quiet=True)
        return super().cleanup(*args, **kwargs)
