import re
import os
import time
import wandb
import datetime
import threading
import pygit2
import tempfile
from dataclasses import dataclass
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
from .mppo_logger import get_logger
from .util import gen_id

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
        source_config: dict,
        vcmi_gym_root: str,
        git_head: str,
        git_is_dirty: bool,
        experiment_name: str,
        resume_from: str | None,
        training_step_duration_s: int,
        hyperparam_mutations: dict,
        wandb_project: str | None,
        wandb_old_run_id: str | None,
        wandb_log_interval_s: int,
        env_runner_keepalive_interval_s: int,
    ):
        self.user = {
            "source_config": source_config,
            "experiment_name": experiment_name,
            "vcmi_gym_root": vcmi_gym_root,
            "git_head": git_head,
            "git_is_dirty": git_is_dirty,
            "resume_from": resume_from,
            "training_step_duration_s": training_step_duration_s,
            "hyperparam_mutations": hyperparam_mutations,
            "wandb_project": wandb_project,
            "wandb_old_run_id": wandb_old_run_id,
            "wandb_log_interval_s": wandb_log_interval_s,
            "env_runner_keepalive_interval_s": env_runner_keepalive_interval_s,
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
        assert self.user["training_step_duration_s"] > 0
        assert self.user["wandb_log_interval_s"] > 0
        assert self.user["wandb_log_interval_s"] <= self.user["training_step_duration_s"]
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
        print("SELF.TRIAL_ID: %s" % self.trial_id)
        assert hasattr(config, "_master_config"), "call .master_config() on the MPPO_Config first"
        self.ns = MPPO_Algorithm.Namespace(
            master_config=config._master_config,
            log_interval=config.user["wandb_log_interval_s"]
        )

        # self.logger.debug("*** SETUP: %s" % json_dumps_safer(config.to_dict()))
        super().setup(config)
        self._wandb_init()
        self.logger = get_logger(self.ns.run_id, "DEBUG")  # *after* wandb init
        self.logger.info("Logger initialized")

        # Must be *after* super().setup()
        self._wandb_add_watch()
        self._wandb_log_hyperparams()

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

        started_at = time.time()
        logged_at = started_at
        training_step_duration_s = self.config.user["training_step_duration_s"]
        wandb_log_interval_s = self.config.user["wandb_log_interval_s"]
        keepalive_interval_s = self.config.user["env_runner_keepalive_interval_s"]

        with EnvRunnerKeepalive(self.eval_env_runner_group, keepalive_interval_s, self.logger):
            # XXX: will this fixed-time step be problematic with ray SPOT
            #       instances where different runners may run with different
            #       speeds?
            while True:
                if os.getenv("DEBUG"):
                    import ipdb; ipdb.set_trace()  # noqa
                result = super().training_step()
                now = time.time()

                # Call custom MPPO-specific callback once every N seconds
                if (now - logged_at) > wandb_log_interval_s:
                    # NOTE: do not call self.metrics.reduce (resets values)
                    self.callbacks.on_train_subresult(self, result)
                    logged_at = now

                if (now - started_at) > training_step_duration_s:
                    break

        return result

    @override(PPO)
    def evaluate(self, *args, **kwargs):
        keepalive_interval_s = self.config.user["env_runner_keepalive_interval_s"]
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
        old_run_id = self.config.user["wandb_old_run_id"]

        if old_run_id:
            assert re.match(r"^[0-9a-z]+_[0-9]+$", old_run_id), f"bad id to resume: {old_run_id}"
            run_id = "%s_%s" % (old_run_id.split("_")[0], self.trial_id.split("_")[1])
            print("Will resume run as id %s (Trial ID is %s)" % (run_id, self.trial_id))
        else:
            run_id = gen_id() if self.trial_id == "default" else self.trial_id
            print("Will start new run %s" % run_id)

        run_name = self.trial_name
        if self.trial_name == "default":
            run_name = f"{datetime.datetime.now().isoformat()}-debug-{run_id}"
        else:
            run_name = "T%d" % int(self.trial_name.split("_")[-1])

        self.ns.run_id = run_id
        self.ns.run_name = run_name

        if self.config.user["wandb_project"]:
            wandb.init(
                project=self.config.user["wandb_project"],
                group=self.config.user["experiment_name"],
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

        for k, v in self.config.user["hyperparam_mutations"].items():
            if k == "env_config":
                for k1, v1 in v.items():
                    assert k1 in self.config.env_config, f"{k1} in self.config.env_config"
                    assert "/" not in k1
                    to_log[f"params/env.{k1}"] = self.config.env_config[k1]
            else:
                assert hasattr(self.config, k), f"hasattr(self.config, {k})"
                assert "/" not in k
                to_log[f"params/{k}"] = getattr(self.config, k)

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
        assert head == self.config.user["git_head"]

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
            # self.logger.debug(f"Sleeping {self.interval}s ...")
            self.event.wait(timeout=self.interval)
            # Render as simple ping mechanism (unwrap to bypass OrderEnforcing)
            self.runner_group.foreach_worker(
                func=lambda w: [any(env.unwrapped.render()) for env in w.env.envs],
                timeout_seconds=10,  # should be instant
                local_env_runner=False,
            )

    def _debug_loop(self):
        def debug(w):
            import ipdb; ipdb.set_trace()  # noqa
        self.runner_group.foreach_worker(func=debug)
