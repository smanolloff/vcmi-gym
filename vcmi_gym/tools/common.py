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

import math
import traceback
import re
import glob
import os
import yaml
import numpy as np
import string
import random
import time
import importlib
import gymnasium as gym
import wandb

import concurrent.futures
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .. import VcmiEnv


class Clock:
    """A better alternative to pygame.Clock for our use-case"""

    def __init__(self, fps):
        self.fps = fps
        self.min_interval = 1 / fps
        self.last_tick_at = time.time()

    def tick(self):
        tick_at = time.time()
        interval = tick_at - self.last_tick_at
        sleep_for = self.min_interval - interval

        if sleep_for > 0:
            time.sleep(sleep_for)
        elif self.fps < 0:
            # manual mode
            input("Press Enter to continue...")

        self.last_tick_at = tick_at + sleep_for


def exp_decay_fn(initial_value, final_value, decay_fraction, n_decays):
    assert initial_value > final_value
    assert final_value > 0
    assert decay_fraction >= 0 and decay_fraction <= 1
    assert n_decays > 0

    # https://stackoverflow.com/a/56400152
    multiplier = math.exp(math.log(final_value / initial_value) / n_decays)
    const_fraction = 1 - decay_fraction
    milestones = [
        const_fraction + decay_fraction * step / (n_decays)
        for step in range(0, n_decays)
    ]

    # [0.9, 0.8, ... 0.1] for 9 decays
    milestones.reverse()

    def func(progress_remaining: float) -> float:
        value = initial_value
        for m in milestones:
            if progress_remaining < m:
                value *= multiplier
                if value < final_value:
                    break
            else:
                break
        return value

    return func


def lin_decay_fn(initial_value, final_value, decay_fraction):
    assert initial_value > final_value
    assert final_value > 0
    assert decay_fraction >= 0 and decay_fraction <= 1

    def func(progress_remaining: float) -> float:
        return max(0, 1 + (initial_value * progress_remaining - 1) / decay_fraction)

    return func


def lr_from_schedule(schedule):
    r_float = r"\d+(?:\.\d+)?"
    r_const = rf"^const_({r_float})$"
    r_lin = rf"^lin_decay_({r_float})_({r_float})_({r_float})$"
    r_exp = rf"^exp_decay_({r_float})_({r_float})_({r_float})(?:_(\d+))$"

    m = re.match(r_const, schedule)
    if m:
        return float(m.group(1))

    m = re.match(r_lin, schedule)
    if m:
        return lin_decay_fn(
            initial_value=float(m.group(1)),
            final_value=float(m.group(2)),
            decay_fraction=float(m.group(3)),
        )

    m = re.match(r_exp, schedule)
    if m:
        return exp_decay_fn(
            initial_value=float(m.group(1)),
            final_value=float(m.group(2)),
            decay_fraction=float(m.group(3)),
            n_decays=int(m.group(4)) if len(m.groups()) > 4 else 10,
        )

    print("Invalid config value for learner_lr_schedule: %s" % schedule)
    exit(1)


def out_dir_from_template(tmpl, seed, run_id, allow_existing=False):
    out_dir = tmpl.format(seed=seed, run_id=run_id)

    if os.path.exists(out_dir) and not allow_existing:
        raise Exception("Output directory already exists: %s" % out_dir)

    return out_dir


def expand_env_kwargs(env_kwargs):
    env_include_cfg = env_kwargs.pop("__include__", None)

    if env_include_cfg:
        with open(env_include_cfg, "r") as f:
            env_kwargs = yaml.safe_load(f) | env_kwargs

    print(env_kwargs)
    return env_kwargs


def gen_seed():
    return int(np.random.default_rng().integers(2**31))


def gen_id():
    population = string.ascii_lowercase + string.digits
    return str.join("", random.choices(population, k=8))


def measure(func, kwargs):
    t1 = time.time()
    retval = func(**kwargs)
    t2 = time.time()

    return t2 - t1, retval


def save_run_metadata(action, cfg, duration, values):
    cfg = {k: v for k, v in cfg.items() if k != "extras"}
    out_dir = values["out_dir"]
    metadata = dict(values, action=action, config=cfg, duration=duration)

    print("Output directory: %s" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    md_file = os.path.join(out_dir, "metadata.yml")

    with open(md_file, "w") as f:
        f.write(yaml.safe_dump(metadata))


def save_model(out_dir, model):
    os.makedirs(out_dir, exist_ok=True)
    model_file = os.path.join(out_dir, "model.zip")
    model.save(model_file)
    print("Saved model to %s" % model_file)


def register_env(env_kwargs={}, env_wrappers=[]):
    def wrapped_env_creator(**kwargs):
        env = VcmiEnv(**kwargs)

        for wrapper in env_wrappers:
            wrapper_mod = importlib.import_module(wrapper["module"])
            wrapper_cls = getattr(wrapper_mod, wrapper["cls"])
            env = wrapper_cls(env, **wrapper.get("kwargs", {}))

        return env

    envid = "local/VCMI-v0"

    gym.envs.register(id=envid, entry_point=wrapped_env_creator, kwargs=env_kwargs)


def make_absolute(cwd, p):
    if os.path.isabs(p):
        return p
    return f"{cwd}/{p}"


def wandb_init(id, group, notes, config):
    # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/air/integrations/wandb.py#L601-L607
    wandb.init(
        id=id,
        name=id,
        resume="allow",
        reinit=True,
        allow_val_change=True,
        # To disable System/ stats:
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        group=group,
        project="vcmi",
        notes=notes,
        config=config,
        # NOTE: this takes a lot of time, better to have detailed graphs
        #       tb-only (local) and log only most important info to wandb
        # sync_tensorboard=True,
        sync_tensorboard=False,
    )


def extract_dict_value_by_path(data_dict, path):
    keys = path.split('.')  # Split the path into individual keys
    current = data_dict

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        raise Exception("Value not found by path: %s (%s)" % (path, traceback.format_exc()))


def maybe_save(t, model, out_dir, save_every, max_saves):
    now = time.time()

    if t is None:
        return now

    if t + save_every > now:
        return t

    os.makedirs(out_dir, exist_ok=True)
    model_file = os.path.join(out_dir, "model-%d.zip" % now)
    print("Saving %s" % model_file)
    model.save(model_file)

    # save file retention (keep latest N saves)
    files = sorted(
        glob.glob(os.path.join(out_dir, "model-[0-9]*.zip")),
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()),
        reverse=True
    )

    for file in files[max_saves:]:
        print("Deleting %s" % file)
        os.remove(file)

    return now


def make_vec_env_parallel(j, env_creator, n_envs, monitor_kwargs):
    def initenv():
        env = env_creator()
        env = Monitor(env, filename=None, **monitor_kwargs)
        return env

    with concurrent.futures.ThreadPoolExecutor(max_workers=j) as executor:
        futures = [executor.submit(initenv) for _ in range(n_envs)]
        results = [future.result() for future in futures]

    funcs = [lambda x=x: x for x in results]

    vec_env = DummyVecEnv(funcs)
    vec_env.seed()
    return vec_env
