import math
import re
import os
import yaml


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
    const_fraction = 1 - decay_fraction

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


def out_dir_from_template(tmpl, seed, run_id):
    out_dir = tmpl.format(seed=seed, run_id=run_id)

    if os.path.exists(out_dir):
        raise Exception("Output directory already exists: %s" % out_dir)

    return out_dir


def expand_env_kwargs(env_kwargs):
    env_include_cfg = env_kwargs.pop("__include__", None)

    if env_include_cfg:
        with open(env_include_cfg, "r") as f:
            env_kwargs = yaml.safe_load(f) | env_kwargs

    return env_kwargs
