import os
import copy
import re
import statistics
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import ray.tune
import ray.train
import wandb
from datetime import datetime

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
        print("-- %s [%s I=%d] %s" % (datetime.now().isoformat(), self.trial_name, self.iteration, msg))

    @staticmethod
    def default_resource_request(_config):
        return ray.tune.execution.placement_groups.PlacementGroupFactory([{"CPU": 1}])

    # XXX: in case of SIGTERM/SIGINT, ray does not wait for cleanup to finish
    #      during regular perturbation, it waits though
    @debuglog
    def cleanup(self):
        self.model.env.close()
        wandb.finish(quiet=True)

    @debuglog
    def setup(self, cfg, initargs):
        self.rollouts_per_iteration = initargs["config"]["rollouts_per_iteration"]
        self.rollouts_per_role = initargs["config"]["rollouts_per_role"]
        self.maps_per_iteration = initargs["config"]["maps_per_iteration"]
        self.logs_per_iteration = initargs["config"]["logs_per_iteration"]
        self.hyperparam_bounds = initargs["config"]["hyperparam_bounds"]
        self.initial_checkpoint = initargs["config"]["initial_checkpoint"]
        self.experiment_name = initargs["experiment_name"]
        self.all_params = copy.deepcopy(initargs["config"]["all_params"])

        self._wandb_init(self.experiment_name, initargs["config"])

        self.sb3_callback = SB3Callback()
        self.leaf_keys = self._get_leaf_hyperparam_keys(self.hyperparam_bounds)
        self.reset_config(cfg)

        assert self.rollouts_per_iteration % self.logs_per_iteration == 0
        self.log_interval = self.rollouts_per_iteration // self.logs_per_iteration

        # Ensure both roles get equal amount of rollouts
        assert self.rollouts_per_iteration % self.rollouts_per_role == 0
        assert (self.rollouts_per_iteration // self.rollouts_per_role) % 2 == 0

        # Ensure there's equal amount of logs for each role
        assert self.rollouts_per_role % self.log_interval == 0

        assert self.rollouts_per_iteration % self.maps_per_iteration == 0
        self.rollouts_per_map = self.rollouts_per_iteration // self.maps_per_iteration

        assert self.rollouts_per_map % self.rollouts_per_role == 0
        assert (self.rollouts_per_map // self.rollouts_per_role) % 2 == 0

        # The model and env should be inited here
        # However, self.iteration is wrong (always 0) in setup()
        # => do lazy init in step()
        self.model = None
        self.initial_side = "attacker"

    @debuglog
    def reset_config(self, cfg):
        cfg = deepmerge(copy.deepcopy(self.all_params), cfg)
        self.cfg = self._fix_floats(copy.deepcopy(cfg), self.hyperparam_bounds)
        steps_per_rollout = cfg["learner_kwargs"]["n_steps"]
        self.total_timesteps_per_iteration = steps_per_rollout * self.rollouts_per_iteration
        self.total_timesteps_per_role = steps_per_rollout * self.rollouts_per_role
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
        self.model = self._model_checkpoint_load(f)

    @debuglog
    def step(self):
        if not self.model:
            # lazy init model - see note in setup()
            self.model = self._model_init()

        wlog = self._get_perturbed_config()
        wlog["trial/iteration"] = self.iteration

        # Commit will be done by the callback's first log
        wandb.log(wlog, commit=False)

        old_rollouts = self.sb3_callback.rollouts
        rollouts_this_iteration = 0
        side = self.initial_side
        ep_rew_means = []
        while rollouts_this_iteration < self.rollouts_per_iteration:
            if len(ep_rew_means) > 0:
                side = "attacker" if len(ep_rew_means) % 2 == 0 else "defender"
                self.model.env.close()
                self.model.env = self._venv_init(side, rollouts_this_iteration)
                self.model.env.reset()

            self.model.learn(
                total_timesteps=self.total_timesteps_per_role,
                log_interval=self.log_interval,
                reset_num_timesteps=(self.iteration == 0 and rollouts_this_iteration == 0),
                progress_bar=False,
                callback=self.sb3_callback
            )

            rollouts_this_iteration += self.rollouts_per_role
            ep_rew_means.append(self.sb3_callback.ep_rew_mean)

        diff_rollouts = self.sb3_callback.rollouts - old_rollouts
        iter_rollouts = self.rollouts_per_iteration

        assert diff_rollouts == iter_rollouts, f"expected {iter_rollouts}, got: {diff_rollouts}"

        # TODO: add net_value to result (to be saved as checkpoint metadata)
        report = {"rew_mean": statistics.mean(ep_rew_means)}
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

    def _venv_init(self, role, rollouts_this_iteration=0):
        env_kwargs = dict(self.cfg["env_kwargs"], attacker="StupidAI", defender="StupidAI")
        env_kwargs[role] = "MMAI_USER"
        mpool = self.all_params["map_pool"]

        offset = self.all_params["map_pool_offset_idx"]
        offset += self.maps_per_iteration * self.iteration

        mid = (offset + rollouts_this_iteration // self.rollouts_per_map) % len(mpool)
        env_kwargs["mapname"] = "ai/generated/%s" % (mpool[mid])
        mapnum = int(re.match(r".+?([0-9]+)\.vmap", env_kwargs["mapname"]).group(1))

        env_kwargs["actions_log_file"] = f"/tmp/{self.trial_id}-actions.log"

        wandb.log(
            # 0=attacker, 1=defender
            {"mapnum": mapnum, "role": ["attacker", "defender"].index(role)},
            commit=False
        )

        self.log("Env kwargs: %s" % env_kwargs)

        return make_vec_env(
            "VCMI-v0",
            n_envs=1,
            env_kwargs=env_kwargs,
            monitor_kwargs={"info_keywords": InfoDict.ALL_KEYS},
        )

    def _model_internal_load(self, f, venv, **learner_kwargs):
        return PPO.load(f, env=venv, **learner_kwargs)

    def _model_internal_init(self, venv, **learner_kwargs):
        return PPO(env=venv, **learner_kwargs)

    def _model_init(self):
        venv = self._venv_init(self.initial_side)

        # at init, we are the origin
        origin = int(self.trial_id.split("_")[1])
        # => int("00002") => 2

        wandb.log({"trial/checkpoint_origin": origin}, commit=False)

        if self.initial_checkpoint:
            self.log("Load %s (initial)" % self.initial_checkpoint)
            return self._model_internal_load(self.initial_checkpoint, venv, **self.cfg["learner_kwargs"])

        return self._model_internal_init(venv, **self.cfg["learner_kwargs"])

    def _model_checkpoint_load(self, f):
        venv = self._venv_init(self.initial_side)

        # Checkpoint tracking: log the trial ID of the checkpoint we are restoring now
        relpath = re.match(fr".+/{self.experiment_name}/(.+)", f).group(1)
        # => "6e59d_00004/checkpoint_000038/model.zip"

        origin = int(re.match(r".+?_(\d+)/.+", relpath).group(1))
        # => int("00004") => 4

        wandb.log({"trial/checkpoint_origin": origin}, commit=False)
        self.log("Load %s (origin: %d)" % (relpath, origin))

        return self._model_internal_load(f, venv, **self.cfg["learner_kwargs"])

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
            if name == "clip_range":
                # clip_range is stored as a constant_fn callable
                params[f"config/{name}"] = self.model.clip_range(1)
            elif hasattr(self.model, name):
                params[f"config/{name}"] = getattr(self.model, name)
            elif hasattr(env, name):
                params[f"config/{name}"] = getattr(env, name)
            else:
                raise Exception("Could not find value for %s" % name)

        return params

    # Needed as Tune passes `float32` objects,
    # but SB3 expects regular `float` objects (or int for some params)
    # Also needed for PBT as values can go out of bounds:
    # https://github.com/ray-project/ray/issues/5035
    def _fix_floats(self, cfg, hyperparam_bounds):
        for key, value in hyperparam_bounds.items():
            if isinstance(value, dict):
                assert key in cfg and isinstance(cfg[key], dict)
                self._fix_floats(cfg[key], value)
            else:
                if key in ["n_epochs", "n_steps", "reward_clip_mod"]:
                    cfg[key] = int(cfg[key])
                else:
                    cfg[key] = float(cfg[key])

                # XXX: PBT only
                if hasattr(value, "lower"):
                    cfg[key] = min(cfg[key], value.upper)
                    cfg[key] = max(cfg[key], value.lower)

        return cfg
