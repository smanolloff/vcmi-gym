import os
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import ray.tune
import ray.train
import wandb

from .sb3_callback import SB3Callback
from ... import InfoDict

DEBUG = False


def debuglog(func):
    if not DEBUG:
        return func

    def wrapper(*args, **kwargs):
        PPOTrainer.log("[%s] Start: %s (args=%s, kwargs=%s)\n" % (args[0].trial_name, func.__name__, args[1:], kwargs))
        result = func(*args, **kwargs)
        PPOTrainer.log("[%s] End: %s (return %s)" % (args[0].trial_name, func.__name__, result))
        return result

    return wrapper


# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Trainable.html
class PPOTrainer(ray.tune.Trainable):
    logfile = None

    @classmethod
    def log(cls, msg):
        # if not cls.logfile:
        #     cls.logfile = tempfile.NamedTemporaryFile(mode="w")
        #     print("-- %s [%s] Logfile: %s\n" % (time.time(), os.getpid(), cls.logfile.name))

        # cls.logfile.write("%s\n" % msg)
        # cls.logfile.flush()
        print(msg)

    @staticmethod
    def default_resource_request(_config):
        return ray.tune.execution.placement_groups.PlacementGroupFactory([{"CPU": 1}])

    # XXX: in case of SIGTERM/SIGINT, ray does not wait for cleanup to finish
    #      during regular perturbation, it waits though
    @debuglog
    def cleanup(self):
        self.venv.close()
        wandb.finish(quiet=True)

    @debuglog
    def setup(self, cfg, initargs):
        self.rollouts_per_iteration = initargs["config"]["rollouts_per_iteration"]
        self.logs_per_iteration = initargs["config"]["logs_per_iteration"]
        self.hyperparam_bounds = initargs["config"]["hyperparam_bounds"]

        self._wandb_init(initargs["experiment_name"], initargs["config"])

        self.sb3_callback = SB3Callback()
        self.leaf_keys = self._get_leaf_hyperparam_keys(self.hyperparam_bounds)
        self.reset_config(cfg)

        self.venv = make_vec_env(
            "VCMI-v0",
            n_envs=1,
            env_kwargs=self.cfg["env_kwargs"],
            monitor_kwargs={"info_keywords": InfoDict.ALL_KEYS},
        )

        self.model = PPO(env=self.venv, **self.cfg["learner_kwargs"])

        assert self.rollouts_per_iteration % self.logs_per_iteration == 0
        self.log_interval = self.rollouts_per_iteration // self.logs_per_iteration

    @debuglog
    def reset_config(self, cfg):
        self.cfg = self._fix_floats(copy.deepcopy(cfg), self.hyperparam_bounds)
        steps_per_rollout = cfg["learner_kwargs"]["n_steps"]
        self.total_timesteps = steps_per_rollout * self.rollouts_per_iteration
        return True

    @debuglog
    def save_checkpoint(self, checkpoint_dir):
        f = os.path.join(checkpoint_dir, "model.zip")
        self.model.save(f)
        return checkpoint_dir

    @debuglog
    def load_checkpoint(self, checkpoint_dir):
        f = os.path.join(checkpoint_dir, "model.zip")
        self.model = PPO.load(f, env=self.venv, **self.cfg["learner_kwargs"])

    @debuglog
    def step(self):
        self.model.env.reset()

        wlog = self._get_perturbed_config()
        wlog["iteration"] = self.iteration

        # Commit will be done by the callback's first log
        wandb.log(wlog, commit=False)

        old_rollouts = self.sb3_callback.rollouts

        self.model.learn(
            total_timesteps=self.total_timesteps,
            log_interval=self.log_interval,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=self.sb3_callback
        )

        diff_rollouts = self.sb3_callback.rollouts - old_rollouts
        iter_rollouts = self.rollouts_per_iteration

        assert diff_rollouts == iter_rollouts, f"expected {iter_rollouts}, got: {diff_rollouts}"

        report = {"rew_mean": self.sb3_callback.ep_rew_mean}
        return report

    @debuglog
    def log_result(self, result):
        # Perform the last log for the teration
        # 'rew_mean': -10411.09756097561
        # 'training_iteration': 5
        # 'trial_id': '874e0_00001'
        # 'date': '2023-11-24_15-58-42'
        # 'timestamp': 1700834322
        # 'time_this_iter_s': 2.4910812377929688
        # 'time_total_s': 12.411049842834473
        # 'pid': 43232
        # 'hostname': 'Simeons-MacBook-Pro.local'
        # 'node_ip': '127.0.0.1'
        # 'config': {...}
        # 'time_since_restore': 12.411049842834473
        # 'iterations_since_restore': 5
        assert self.sb3_callback.uncommitted_log, "Expected uncommitted sb3 cb log"
        wandb.log({"trial/time_this_iter_s": result["time_this_iter_s"]})

    #
    # private
    #

    def _wandb_init(self, experiment_name, config):
        # print("[%s] INITWANDB: PID: %s, trial_id: %s" % (time.time(), os.getpid(), trial_id))

        # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/air/integrations/wandb.py#L601-L607
        wandb.init(
            id=self.trial_id,
            name="PB2_%s" % self.trial_name.split("_")[-1],
            resume="allow",
            reinit=True,
            allow_val_change=True,
            # To disable System/ stats:
            # settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
            group=experiment_name,
            project=config["wandb_project"],
            config=config,
            # NOTE: this takes a lot of time, better to have detailed graphs
            #       tb-only (local) and log only most important info to wandb
            # sync_tensorboard=True,
            sync_tensorboard=False,
        )
        # print("[%s] DONE WITH INITWANDB" % time.time())

    def _get_leaf_hyperparam_keys(self, data):
        leaf_keys = []

        for key, value in data.items():
            if isinstance(value, list):
                leaf_keys.append(key)
            else:
                leaf_keys.extend(self._get_leaf_hyperparam_keys(value))

        return leaf_keys

    # NOTE: This could be extracted from self.cfg
    #       However, extracting those values from the objects is more accurate
    #       and has proved useful in the past
    def _get_perturbed_config(self):
        params = {}
        env = self.model.env.envs[0].unwrapped

        for name in self.leaf_keys:
            if hasattr(self.model, name):
                params[f"config/{name}"] = getattr(self.model, name)
            elif hasattr(env, name):
                params[f"config/{name}"] = getattr(env, name)
            else:
                raise Exception("Could not find value for %s" % name)

        return params

    # Needed as Tune passes `float32` objects,
    # but SB3 expects regular `float` objects
    def _fix_floats(self, cfg, hyperparam_bounds):
        for key, value in hyperparam_bounds.items():
            if isinstance(value, dict):
                assert key in cfg and isinstance(cfg[key], dict)
                self._fix_floats(cfg[key], value)
            else:
                cfg[key] = float(cfg[key])
        return cfg
