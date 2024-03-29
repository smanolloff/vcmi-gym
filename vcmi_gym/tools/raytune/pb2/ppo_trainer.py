# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import copy
import re
import statistics
import glob
import random
import threading
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
import ray.tune
import ray.train
import wandb
import torch.optim
from datetime import datetime

from ..sb3_callback import SB3Callback
from ..wandb_init import wandb_init
from stable_baselines3.common.save_util import load_from_zip_file
# from .... import InfoDict
import vcmi_gym
from vcmi_gym import InfoDict
from vcmi_gym.tools.common import make_vec_env_parallel

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
        self.iteration_steps = initargs["config"]["iteration_steps"]
        self.rollouts_per_iteration_step = initargs["config"]["rollouts_per_iteration_step"]
        self.rollouts_per_log = initargs["config"]["rollouts_per_log"]
        self.n_envs = initargs["config"]["n_envs"]
        self.hyperparam_bounds = initargs["config"]["hyperparam_bounds"]
        self.initial_checkpoint = initargs["config"]["initial_checkpoint"]
        self.experiment_name = initargs["experiment_name"]
        self.all_params = copy.deepcopy(initargs["config"]["all_params"])

        self._wandb_init(self.experiment_name, initargs["config"])

        self.sb3_callback = SB3Callback()
        self.leaf_keys = self._get_leaf_hyperparam_keys(self.hyperparam_bounds)
        self.reset_config(cfg)

        assert self.rollouts_per_iteration_step % self.rollouts_per_log == 0

        assert self.n_envs % 2 == 0 or self.n_envs == 1

        # The model and env should be inited here
        # However, self.iteration is wrong (always 0) in setup()
        # => do lazy init in step()
        self.model = None
        self.initial_side = "attacker"

    @debuglog
    def reset_config(self, cfg):
        cfg = deepmerge(copy.deepcopy(self.all_params), cfg)
        self.cfg = self._fix_floats(copy.deepcopy(cfg), self.hyperparam_bounds)

        # SB3 increments its n_steps counter by 1 on call to venv.step(),
        # regardless of n_envs (=> n_steps != timesteps)
        n_steps_per_rollout = cfg["learner_kwargs"]["n_steps"]
        timesteps_per_rollout = n_steps_per_rollout * self.n_envs

        self.total_timesteps_per_iteration_step = timesteps_per_rollout * self.rollouts_per_iteration_step
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
        iteration_step = 0
        ep_rew_means = []

        while iteration_step < self.iteration_steps:
            self.log("Iteration step: %d/%d (%d rollouts per step)" % (
                iteration_step,
                self.iteration_steps,
                self.rollouts_per_iteration_step,
            ))

            if iteration_step > 0:
                self.model.env.close()
                self.model.env = self._venv_init(iteration_step)
                self.model.env.reset()

            self.model.learn(
                total_timesteps=self.total_timesteps_per_iteration_step,
                log_interval=self.rollouts_per_log,
                reset_num_timesteps=(self.iteration == 0 and iteration_step == 0),
                progress_bar=False,
                callback=self.sb3_callback
            )

            diff_rollouts = self.sb3_callback.rollouts - (iteration_step * self.rollouts_per_iteration_step)
            assert diff_rollouts == self.rollouts_per_iteration_step, f"expected {self.rollouts_per_iteration_step}, got: {diff_rollouts}"  # noqa: E501
            iteration_step += 1
            ep_rew_means.append(self.sb3_callback.ep_rew_mean)

        diff_rollouts = self.sb3_callback.rollouts - old_rollouts
        iter_rollouts = self.rollouts_per_iteration_step * self.iteration_steps

        assert diff_rollouts == iter_rollouts, f"expected {iter_rollouts}, got: {diff_rollouts}"

        # TODO: add net_value to result (to be saved as checkpoint metadata)
        report = {"rew_mean": statistics.mean(ep_rew_means)}
        return report

    @debuglog
    def log_result(self, result):
        # assert self.sb3_callback.uncommitted_log
        # assert self.sb3_callback.wdb_tables
        # wdb_log = dict(self.sb3_callback.wdb_tables)
        # wdb_log["trial/time_this_iter_s"] = result["time_this_iter_s"]
        # wandb.log(wdb_log)
        pass

    #
    # private
    #

    def _wandb_init(self, experiment_name, config):
        wandb_init(self.trial_id, self.trial_name, experiment_name, config)

    def _venv_init(self, iteration_step=0):
        mappath = "/Users/simo/Library/Application Support/vcmi/Maps"
        all_maps = glob.glob("%s/%s" % (mappath, self.all_params["mapmask"]))
        all_maps = [m.replace("%s/" % mappath, "") for m in all_maps]
        all_maps.sort()

        if self.n_envs == 1:
            n_maps = 1
        else:
            assert self.n_envs % 2 == 0
            assert self.n_envs <= len(all_maps) * 2
            n_maps = self.n_envs // 2

        if self.all_params["randomize_maps"]:
            maps = random.sample(all_maps, n_maps)
        else:
            iteration = self.iteration_steps * self.iteration + iteration_step
            i = (n_maps * iteration) % len(all_maps)
            new_i = (i + n_maps) % len(all_maps)
            wandb.log({"map_offset": i}, commit=False)

            if new_i > i:
                maps = all_maps[i:new_i]
            else:
                maps = all_maps[i:] + all_maps[:new_i]

            assert len(maps) == n_maps

        pairs = [[("attacker", m), ("defender", m)] for m in maps]
        pairs = [x for y in pairs for x in y]  # aka. pairs.flatten(1)...
        state = {"n": 0}
        lock = threading.RLock()

        def env_creator(**_env_kwargs):
            with lock:
                assert state["n"] < self.n_envs
                role, mapname = pairs[state["n"]]
                # logfile = f"/tmp/{run_id}-env{state['n']}-actions.log"
                logfile = None

                env_kwargs2 = dict(
                    self.cfg["env_kwargs"],
                    mapname=mapname,
                    attacker="StupidAI",
                    defender="StupidAI",
                    actions_log_file=logfile
                )

                env_kwargs2[role] = "MMAI_USER"
                print("Env kwargs (env.%d): %s" % (state["n"], env_kwargs2))
                state["n"] += 1

            return vcmi_gym.VcmiEnv(**env_kwargs2)

        return make_vec_env_parallel(
            min(self.n_envs, 8),
            env_creator,
            n_envs=self.n_envs,
            monitor_kwargs={"info_keywords": InfoDict.ALL_KEYS},
        )

    # PPO.load() compares the policy_kwargs at the top-level
    # with the ones stored in the .zip file and will blow up on mismatch.
    #
    # * if the network arch itself is different => error
    #   => net can NOT change
    # * if the optimizer class is different => danger (state load error?)
    #   => optimizer class can NOT change
    # * if the optimizer kwargs are different => nothing happens
    #   (SB3 loads entire optimizer state later, overwriting kwargs)
    #   => optimizer kwargs have NO effect
    #
    # => use only learner_kwargs, and explicitly update optimizer later
    def _learner_kwargs_for_load(self):
        return dict(self.cfg["learner_kwargs"])

    def _learner_kwargs_for_init(self):
        policy_kwargs = {
            "net_arch": self.cfg["net_arch"],
            "activation_fn": getattr(torch.nn, self.cfg["activation"]),
        }

        # Any custom features extractor is assumed to be a VcmiCNN-type policy
        if self.cfg["features_extractor"]:
            policy_kwargs["features_extractor_class"] = getattr(vcmi_gym, self.cfg["features_extractor"]["class_name"])
            policy_kwargs["features_extractor_kwargs"] = self.cfg["features_extractor"]["kwargs"]

        if self.cfg["optimizer"]:
            policy_kwargs["optimizer_class"] = getattr(torch.optim, self.cfg["optimizer"]["class_name"])
            policy_kwargs["optimizer_kwargs"] = self.cfg["optimizer"]["kwargs"]

        return dict(
            self.cfg["learner_kwargs"],
            policy="MlpPolicy",
            policy_kwargs=policy_kwargs
        )

    def _model_internal_load(self, f, venv, **learner_kwargs):
        return PPO.load(f, env=venv, **learner_kwargs)

    def _model_internal_init(self, venv, **learner_kwargs):
        return PPO(env=venv, **learner_kwargs)

    def _features_extractor_init(self, model):
        features_extractor_load_file = self.all_params["features_extractor_load_file"]
        if not features_extractor_load_file:
            return model

        self.log("Loading features extractor from %s" % features_extractor_load_file)
        features_extractor_load_file_type = self.all_params["features_extractor_load_file_type"]

        if features_extractor_load_file_type == "sb3":
            _data, params, _pytorch_variables = load_from_zip_file(features_extractor_load_file)
            prefix = "features_extractor."
            features_extractor_params = dict(
                (k.removeprefix(prefix), v) for (k, v) in params["policy"].items() if k.startswith(prefix)
            )
        elif features_extractor_load_file_type == "params":
            autoencoder_encoder_params = torch.load(features_extractor_load_file)
            features_extractor_params = dict(
                (f"network.{k}", v) for (k, v) in autoencoder_encoder_params.items()
            )
        elif features_extractor_load_file_type == "model":
            model = torch.load(features_extractor_load_file)
            features_extractor_params = model.state_dict()
        else:
            raise Exception("Unexpected features_extractor_load_file_type: %s" % features_extractor_load_file_type)

        model.policy.features_extractor.load_state_dict(features_extractor_params, strict=True)

        return model

    def _maybe_freeze_features_extractor(self, model):
        if not self.all_params["features_extractor_freeze"]:
            return

        self.log("Freezing features extractor params")
        for param in model.policy.features_extractor.parameters():
            param.requires_grad = False

    def _model_init(self):
        # at init, we are the origin
        origin = int(self.trial_id.split("_")[1])
        # => int("00002") => 2

        wandb.log({"trial/checkpoint_origin": origin}, commit=False)
        model = None

        if self.initial_checkpoint:
            self.log("Load %s (initial)" % self.initial_checkpoint)
            model = self._model_load(self.initial_checkpoint)
        else:
            venv = self._venv_init()
            model = self._model_internal_init(venv, **self._learner_kwargs_for_init())
            model = self._features_extractor_init(model)
            self._maybe_freeze_features_extractor(model)

        return model

    def _model_load(self, f):
        venv = self._venv_init()
        model = self._model_internal_load(f, venv, **self._learner_kwargs_for_load())

        # Need to explicitly update optimizer settings after load
        optimizer = self.cfg.get("optimizer", {})

        if optimizer:
            klass = model.policy.optimizer.__class__.__name__
            # XXX: SB3 loads optimizer state from ZIP
            # There may be issues if the class is different
            assert optimizer["class_name"] == klass, "updating optimizer class is not supported"

            for (k, v) in optimizer.get("kwargs", {}).items():
                model.policy.optimizer.param_groups[0][k] = v

        self._maybe_freeze_features_extractor(model)

        return model

    def _model_checkpoint_load(self, f):
        # Checkpoint tracking: log the trial ID of the checkpoint we are restoring now
        relpath = re.match(fr".+/{self.experiment_name}/(.+)", f).group(1)
        # => "6e59d_00004/checkpoint_000038/model.zip"

        origin = int(re.match(r".+?_(\d+)/.+", relpath).group(1))
        # => int("00004") => 4

        wandb.log({"trial/checkpoint_origin": origin}, commit=False)
        self.log("Load %s (origin: %d)" % (relpath, origin))

        return self._model_load(f)

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
                params["config/clip_range"] = self.model.clip_range(1)
            elif name == "net_arch":
                params["config/net_arch"] = repr(self.model.policy.net_arch)
            elif hasattr(self.model, name):
                params[f"config/{name}"] = getattr(self.model, name)
            elif hasattr(env, name):
                params[f"config/{name}"] = getattr(env, name)
            elif name in self.model.policy.optimizer.param_groups[0]:
                value = self.model.policy.optimizer.param_groups[0][name]
                params[f"config/{name}"] = value
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
                elif key in ["net_arch"]:
                    assert isinstance(cfg[key], list)
                else:
                    cfg[key] = float(cfg[key])

                # XXX: PBT only
                if hasattr(value, "lower"):
                    cfg[key] = min(cfg[key], value.upper)
                    cfg[key] = max(cfg[key], value.lower)

        return cfg
