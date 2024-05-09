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

import re
import importlib
import copy
import ray.train
import ray.tune

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


# "n" exponentially distributed numbers in the range [low, high]
def explist(low, high, n=100, dtype=float):
    x = (high/low) ** (1 / (n-1))
    return list(map(lambda i: dtype(low * x**i), range(n)))


# "n" linearly distributed numbers in the range [low, high]
def linlist(low, high, n=100, dtype=float):
    x = (high-low) / (n-1)
    return list(map(lambda i: dtype(low + x*i), range(n)))


# Flatten dict keys: {"a": {"b": 1, "c": 2"}} => ["a.b", "a.c"]
def flattened_dict_keys(d, sep, parent_key=None):
    keys = []
    for k, v in d.items():
        assert sep not in k, "original dict's keys must not contain '%s'" % sep
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            keys.extend(flattened_dict_keys(v, sep, new_key))
        else:
            keys.append(new_key)
    return keys


def deepmerge(a: dict, b: dict, allow_new=True, update_existing=True, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deepmerge(a[key], b[key], allow_new, update_existing, path + [str(key)])
            elif update_existing and a[key] != b[key]:
                a[key] = b[key]
        elif allow_new:
            a[key] = b[key]
        else:
            raise Exception("Key not found: %s" % key)
    return a


# Create a dict with A's keys and B's values (if such are present)
def common_dict(a, b):
    res = {}
    for key, value in a.items():
        if key in b:
            if isinstance(value, dict) and isinstance(b[key], dict):
                res[key] = common_dict(a[key], b[key])
            else:
                res[key] = b[key]
    return res


class TBXDummyCallback(ray.tune.logger.TBXLoggerCallback):
    """ A dummy class to be passed to ray Tuner at init.

    This will trick ray into believing it has a TBX logger already
    and will not create a new, default one.
    I dont want hundreds of tb files created with useless info in my data dir
    """

    def __init__(self):
        pass

    def log_trial_start(self, *args, **kwargs):
        pass

    def log_trial_result(self, *args, **kwargs):
        pass

    def log_trial_end(self, *args, **kwargs):
        pass


def new_tuner(algo, experiment_name, config, scheduler, searcher=None, param_space=None):
    assert algo in ["mppo", "mppo_dna", "mppg"], f"Unsupported algo: {algo}"
    assert re.match(r"^[0-9A-Za-z_-].+$", experiment_name)
    trainable_mod = importlib.import_module("rl.raytune.trainable")
    trainable_cls = getattr(trainable_mod, "Trainable")

    checkpoint_config = ray.train.CheckpointConfig(
        num_to_keep=10,
        # XXX: can't use score as it may delete the *latest* checkpoint
        #      and then fail when attempting to load it after perturb...
        # checkpoint_score_order="max",
        # checkpoint_score_attribute="rew_mean",
    )

    # https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray-train-runconfig
    run_config = ray.train.RunConfig(
        name=experiment_name,
        verbose=False,
        failure_config=ray.train.FailureConfig(max_failures=-1),
        checkpoint_config=checkpoint_config,
        stop={"rew_mean": config["_raytune"]["target_ep_rew_mean"]},
        callbacks=[TBXDummyCallback()],
        # storage_path=results_dir,  # redundant, using TUNE_RESULT_DIR instead
    )

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html#ray.tune.TuneConfig
    tune_config = ray.tune.TuneConfig(
        trial_name_creator=lambda t: t.trial_id,
        trial_dirname_creator=lambda t: t.trial_id,
        scheduler=scheduler,
        reuse_actors=False,  # XXX: False is much safer and ensures no state leaks
        num_samples=config["_raytune"]["population_size"],
        #max_concurrent_trials=config["_raytune"]["max_concurrency"],  # XXX: not working?
        max_concurrent_trials=config["_raytune"]["population_size"],
        search_alg=searcher
    )

    initargs = copy.deepcopy(config)
    initargs["_raytune"]["experiment_name"] = experiment_name
    initargs["_raytune"]["algo"] = algo

    tuner = ray.tune.Tuner(
        trainable=ray.tune.with_parameters(trainable_cls, initargs=initargs),
        run_config=run_config,
        tune_config=tune_config,
        param_space=param_space,
    )

    return tuner


def resume_tuner(algo, resume_path, config):
    assert algo in ["mppo", "mppo_dna", "mppg"], f"Unsupported algo: {algo}"
    trainable_mod = importlib.import_module("rl.raytune.trainable")
    trainable_cls = getattr(trainable_mod, "Trainable")

    initargs = copy.deepcopy(config)
    initargs["_raytune"]["experiment_name"] = resume_path.split("/")[-1]
    initargs["_raytune"]["algo"] = algo

    tuner = ray.tune.Tuner.restore(
        trainable=ray.tune.with_parameters(trainable_cls, initargs=initargs),
        path=resume_path,
    )

    return tuner
