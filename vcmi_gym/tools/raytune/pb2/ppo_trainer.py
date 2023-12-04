import os
import copy
import re
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import ray.tune
import ray.train
import wandb

from ..sb3_callback import SB3Callback
from ..wandb_init import wandb_init
from .... import InfoDict

DEBUG = True


def debuglog(func):
    if not DEBUG:
        return func

    def wrapper(*args, **kwargs):
        args[0].log("Start: %s (args=%s, kwargs=%s)\n" % (func.__name__, args[1:], kwargs))
        result = func(*args, **kwargs)
        args[0].log("End: %s (return %s)" % (func.__name__, result))
        return result

    return wrapper


def deepmerge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deepmerge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                a[key] = b[key]
        else:
            raise Exception("Key not found: %s" % key)
    return a


# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Trainable.html
class PPOTrainer(ray.tune.Trainable):
    # logfile = None

    def log(self, msg):
        # if not cls.logfile:
        #     cls.logfile = tempfile.NamedTemporaryFile(mode="w")
        #     print("-- %s [%s] Logfile: %s\n" % (time.time(), os.getpid(), cls.logfile.name))

        # cls.logfile.write("%s\n" % msg)
        # cls.logfile.flush()
        # print(msg)
        print("[%s I=%d] %s" % (self.trial_name, self.iteration, msg))

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
        self.experiment_name = initargs["experiment_name"]
        self.all_params = copy.deepcopy(initargs["config"]["all_params"])

        self._wandb_init(self.experiment_name, initargs["config"])

        self.sb3_callback = SB3Callback()
        self.leaf_keys = self._get_leaf_hyperparam_keys(self.hyperparam_bounds)
        self.reset_config(cfg)

        self.venv = make_vec_env(
            "VCMI-v0",
            n_envs=1,
            env_kwargs=self.cfg["env_kwargs"],
            monitor_kwargs={"info_keywords": InfoDict.ALL_KEYS},
        )

        if initargs["config"]["initial_checkpoint"]:
            self.model = self._model_init_load(
                f=initargs["config"]["initial_checkpoint"],
                venv=self.venv,
                **self.cfg["learner_kwargs"]
            )
        else:
            self.model = self._model_init(venv=self.venv, **self.cfg["learner_kwargs"])

        assert self.rollouts_per_iteration % self.logs_per_iteration == 0
        self.log_interval = self.rollouts_per_iteration // self.logs_per_iteration

    @debuglog
    def reset_config(self, cfg):
        cfg = deepmerge(copy.deepcopy(self.all_params), cfg)
        self.cfg = self._fix_floats(copy.deepcopy(cfg), self.hyperparam_bounds)
        steps_per_rollout = cfg["learner_kwargs"]["n_steps"]
        self.total_timesteps = steps_per_rollout * self.rollouts_per_iteration
        return True

    @debuglog
    def save_checkpoint(self, checkpoint_dir):
        f = os.path.join(checkpoint_dir, "model.zip")
        self.model.save(f)
        # self.log("(save): self.experiment_name: '%s', f: '%s'" % (self.experiment_name, f))
        # self.log("Saved %s" % f.split(self.experiment_name)[1])
        return checkpoint_dir

    @debuglog
    def load_checkpoint(self, checkpoint_dir):
        f = os.path.join(checkpoint_dir, "model.zip")
        self.model = self._model_checkpoint_load(f, venv=self.venv, **self.cfg["learner_kwargs"])

    @debuglog
    def step(self):
        self.model.env.reset()

        wlog = self._get_perturbed_config()
        wlog["trial/iteration"] = self.iteration

        # Commit will be done by the callback's first log
        wandb.log(wlog, commit=False)

        old_rollouts = self.sb3_callback.rollouts

        self.model.learn(
            total_timesteps=self.total_timesteps,
            log_interval=self.log_interval,
            reset_num_timesteps=(self.iteration == 0),
            progress_bar=False,
            callback=self.sb3_callback
        )

        diff_rollouts = self.sb3_callback.rollouts - old_rollouts
        iter_rollouts = self.rollouts_per_iteration

        assert diff_rollouts == iter_rollouts, f"expected {iter_rollouts}, got: {diff_rollouts}"

        # TODO: add net_value to result (to be saved as checkpoint metadata)
        report = {"rew_mean": self.sb3_callback.ep_rew_mean}
        return report

    @debuglog
    def log_result(self, result):
        assert self.sb3_callback.uncommitted_log
        assert self.sb3_callback.wdb_tables
        wdb_log = dict(self.sb3_callback.wdb_tables)
        wdb_log["trial/time_this_iter_s"] = result["time_this_iter_s"]
        wandb.log(wdb_log)

    #
    # private
    #

    def _wandb_init(self, experiment_name, config):
        wandb_init(self.trial_id, self.trial_name, experiment_name, config)

    def _model_init(self, venv, **learner_kwargs):
        # at init, we are the origin
        origin = int(self.trial_id.split("_")[1])
        # => int("00002") => 2

        wandb.log({"trial/checkpoint_origin": origin}, commit=False)
        return PPO(env=venv, **learner_kwargs)

    def _model_init_load(self, f, venv, **learner_kwargs):
        # at init, we are the origin
        origin = int(self.trial_id.split("_")[1])
        # => int("00002") => 2

        wandb.log({"trial/checkpoint_origin": origin}, commit=False)
        self.log("Load %s (initial)" % f)
        return self._model_load(f, venv=venv, **learner_kwargs)

    def _model_checkpoint_load(self, f, venv, **learner_kwargs):
        # Checkpoint tracking: log the trial ID of the checkpoint we are restoring now
        relpath = re.match(fr".+/{self.experiment_name}/(.+)", f).group(1)
        # => "6e59d_00004/checkpoint_000038/model.zip"

        origin = int(re.match(r".+?_(\d+)/.+", relpath).group(1))
        # => int("00004") => 4

        wandb.log({"trial/checkpoint_origin": origin}, commit=False)
        self.log("Load %s (origin: %d)" % (relpath, origin))
        return self._model_load(f, venv=venv, **learner_kwargs)

    def _model_load(self, f, venv, **learner_kwargs):
        return PPO.load(f, env=venv, **learner_kwargs)

    def _get_leaf_hyperparam_keys(self, data):
        leaf_keys = []

        for key, value in data.items():
            if isinstance(value, dict):
                leaf_keys.extend(self._get_leaf_hyperparam_keys(value))
            else:
                leaf_keys.append(key)

        return leaf_keys

    # NOTE: This could be extracted from self.cfg
    #       However, extracting those values from the objects is more accurate
    #       and has proved useful in the past
    #
    # TODO: `clip_range` is returned as a schedule fn and logged as a string
    #       Need to figure out how to convert it to float
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
    # Also needed for PBT as values can go out of bounds:
    # https://github.com/ray-project/ray/issues/5035
    def _fix_floats(self, cfg, hyperparam_bounds):
        for key, value in hyperparam_bounds.items():
            if isinstance(value, dict):
                assert key in cfg and isinstance(cfg[key], dict)
                self._fix_floats(cfg[key], value)
            else:
                if key == "n_epochs":
                    cfg[key] = int(cfg[key])
                else:
                    cfg[key] = float(cfg[key])

                # XXX: PBT only
                if hasattr(value, "lower"):
                    cfg[key] = min(cfg[key], value.upper)
                    cfg[key] = max(cfg[key], value.lower)

        return cfg
