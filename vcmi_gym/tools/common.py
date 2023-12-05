import math
import re
import os
import yaml
import numpy as np
import string
import random
import time
import importlib
import gymnasium as gym

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


def play_model(env, fps, model, obs):
    terminated = False
    clock = Clock(fps)
    last_errors = 0

    while not terminated:
        if model.__class__.__name__ == "MaskablePPO":
            action, _states = model.predict(obs, action_masks=env.unwrapped.action_masks())
        else:
            action, _states = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

        if info.get("errors", 0) == last_errors:
            clock.tick()

        last_errors = info.get("errors", 0)

    clock.tick()
