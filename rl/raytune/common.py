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

import wandb
from ray.tune.logger import TBXLoggerCallback

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


def wandb_init(trial_id, trial_name, experiment_name, config):
    # print("[%s] INITWANDB: PID: %s, trial_id: %s" % (time.time(), os.getpid(), trial_id))

    # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/air/integrations/wandb.py#L601-L607
    wandb.init(
        id=trial_id,
        name="T%d" % int(trial_name.split("_")[-1]),
        resume="allow",
        reinit=True,
        allow_val_change=True,
        # To disable System/ stats:
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        group=experiment_name,
        project=config["wandb_project"],
        config=config,
        sync_tensorboard=False,
    )
    # print("[%s] DONE WITH INITWANDB" % time.time())


class TBXDummyCallback(TBXLoggerCallback):
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
