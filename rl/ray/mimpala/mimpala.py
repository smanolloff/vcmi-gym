import wandb
import time
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms import IMPALA, IMPALAConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.result import TRIAL_INFO

from ..common import common_config, common_algorithm, common_logger, util
from ..mppo.mppo_rl_module import MPPO_RLModule
from .mimpala_callback import MIMPALA_Callback


class MIMPALA_Config(IMPALAConfig):
    @override(IMPALAConfig)
    def __init__(self):
        super().__init__(algo_class=MIMPALA_Algorithm)
        common_config.init(self, MIMPALA_Callback)

        # !!! This is the API as of ray-2.38.0
        # !!! It *will* change in future releases
        self.training(
            vtrace_clip_rho_threshold=1.0,
            vtrace_clip_pg_rho_threshold=1.0,
            learner_queue_size=3,
            max_requests_in_flight_per_aggregator_worker=2,
            # XXX: these timeouts must remain 0 (sampling is async)
            timeout_s_sampler_manager=0,
            timeout_s_aggregator_manager=0,
            broadcast_interval=1,
            num_aggregation_workers=0,
            entropy_coeff=0.0,
            gamma=0.8,
            grad_clip=5,
            grad_clip_by="global_norm",  # global_norm = nn.utils.clip_grad_norm_(model.parameters)
            lr=0.001,
            minibatch_size=32,
            num_epochs=1,
            train_batch_size_per_learner=500,
            shuffle_batch_per_epoch=True,
            vf_loss_coeff=1.0
        )

        # This is @OldAPIstack, but defaults to 1 and prevents running on Mac
        # (the example with the new API stack also sets it to 0)
        self.resources(num_gpus=0)

    @override(IMPALAConfig)
    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=MPPO_RLModule)

    @property
    @override(IMPALAConfig)
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


class MIMPALA_Algorithm(IMPALA):
    ALGO_NAME = "MIMPALA"

    @classmethod
    @override(IMPALA)
    def get_default_config(cls):
        return MIMPALA_Config()

    @override(IMPALA)
    def __init__(self, *args, **kwargs):
        util.silence_log_noise()
        super().__init__(*args, **kwargs)

    @override(IMPALA)
    def setup(self, config: MIMPALA_Config):
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

    @override(IMPALA)
    def training_step(self):
        # XXX: there's no `on_training_step_start` callback => log this here
        self.wandb_log({"trial/iteration": self.iteration})

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
                self.logger.debug("training_step: call super")
                result = super().training_step()
                self.logger.debug("training_step: done super")
                now = time.time()

                # Call custom MPPO-specific callback once every N seconds
                if (now - logged_at) > wandb_log_interval_s:
                    # NOTE: do not call self.metrics.reduce (resets values)
                    # XXX: If env runner is non-local, IMPALA samples async
                    #      (i.e. result may be empty if no samples are ready)
                    if result:
                        self.callbacks.on_train_subresult(self, result)
                    else:
                        self.logger.debug("no requests, sleep 1s")
                        time.sleep(1)
                    logged_at = now

                self.logger.debug("training_step time left: %ds" % (training_step_duration_s - (now - started_at)))
                if (now - started_at) > training_step_duration_s:
                    break

        return result

    @override(IMPALA)
    def evaluate(self, *args, **kwargs):
        keepalive_interval_s = self.config.user_config.env_runner_keepalive_interval_s
        with common_algorithm.EnvRunnerKeepalive(self.env_runner_group, keepalive_interval_s, self.logger):
            return super().evaluate(*args, **kwargs)

    @override(IMPALA)
    def save_checkpoint(self, checkpoint_dir):
        res = super().save_checkpoint(checkpoint_dir)
        common_algorithm.save_checkpoint(self, checkpoint_dir)
        return res

    # XXX: in case of SIGTERM/SIGINT, ray does not wait for cleanup to finish.
    #      During regular perturbation, though, it does.
    @override(IMPALA)
    def cleanup(self, *args, **kwargs):
        if wandb.run:
            wandb.finish(quiet=True)
        return super().cleanup(*args, **kwargs)
