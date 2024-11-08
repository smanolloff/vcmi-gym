import copy
import random
import string
import pathlib
import logging
import re
import dataclasses
import typing
import gymnasium as gym

# i.e. vcmi-gym root:
vcmigym_root_path = next(p for p in pathlib.Path(__file__).parents if p.name == "vcmi-gym")
data_path = vcmigym_root_path.joinpath("data").absolute()


class LogMessageSuppressor(logging.Filter):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = re.compile(pattern)

    def filter(self, record):
        return not self.pattern.match(record.getMessage())


def silence_log_noise():
    import warnings

    warnings.filterwarnings(
        action="ignore",
        # gym warnings are ANSI-colored
        message=r".+WARN: Overriding environment rllib-single-agent-env-v0 already in registry.",
        category=UserWarning,
        module=r"^gymnasium\."
    )

    import ray.rllib.connectors.connector_pipeline_v2  # noqa
    import ray.rllib.env.env_runner_group  # noqa
    import ray.rllib.utils.deprecation  # noqa
    logging.getLogger("ray.rllib.connectors.connector_pipeline_v2").setLevel(logging.WARNING)
    logging.getLogger("ray.rllib.env.env_runner_group").setLevel(logging.WARNING)
    logging.getLogger("ray.rllib.utils.deprecation").addFilter(
        LogMessageSuppressor(re.escape("DeprecationWarning: `RLModule(config=[RLModuleConfig object])`"))
    )


# "n" exponentially distributed numbers in the range [low, high]
def explist(low, high, n=100, dtype=float):
    x = (high/low) ** (1 / (n-1))
    return list(map(lambda i: dtype(low * x**i), range(n)))


# "n" linearly distributed numbers in the range [low, high]
def linlist(low, high, n=100, dtype=float):
    x = (high-low) / (n-1)
    return list(map(lambda i: dtype(low + x*i), range(n)))


# Merge b into a, optionally preventing new keys; does not mutate inputs
def deepmerge(a: dict, b: dict, in_place=False, allow_new=True, update_existing=True, path=[]):
    if len(path) == 0 and not in_place:
        a = copy.deepcopy(a)

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deepmerge(a[key], b[key], in_place, allow_new, update_existing, path + [str(key)])
            elif update_existing and a[key] != b[key]:
                a[key] = b[key]
        elif allow_new:
            a[key] = b[key]
        else:
            raise Exception("Key not found: %s" % key)
    return a


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


def get_divisors(n, nmin=0):
    divisors = [i for i in range(n // 2, nmin, -1) if n % i == 0]
    return divisors


def gen_id(n=8):
    population = string.ascii_lowercase + string.digits
    return str.join("", random.choices(population, k=n))


# e.g. attrpath="a.b" return obj.a.b
def get_nested_attr(obj, attrpath, sep="."):
    head, _, tail = attrpath.partition(sep)
    obj = getattr(obj, head)
    return get_nested_attr(obj, tail, sep) if tail else obj


def to_abspath(strpath):
    path = pathlib.Path(strpath)
    if path.is_absolute():
        return str(path)
    return str(vcmigym_root_path.joinpath(path))


# Create a dict with A's keys and B's values (if present)
def common_dict(a, b, strict=False):
    res = {}
    for key in a:
        assert isinstance(key, str)

        if key not in b:
            assert not strict, f"b['{key}']: not found"
            continue

        if isinstance(key[a], dict):
            assert isinstance(b[key], dict), f"b['{key}']: expected dict, got: {type(b[key])}"
            res[key] = common_dict(a[key], b[key])
        else:
            assert isinstance(a[key], list), f"a['{key}']: expected list, got: {type(a[key])}"
            assert isinstance(a[key], (int, float, str)), f"a['{key}']: expected int/float/str, got: {type(a[key])}"
            assert key not in res, f"res['{key}']: already exists"
            res[key] = b[key]

    return res


def validate_dataclass_fields(obj):
    # Get type hints for all fields in the dataclass
    type_hints = typing.get_type_hints(obj.__class__)

    # Validate each field's type
    for field in dataclasses.fields(obj):
        field_name = field.name
        field_value = getattr(obj, field_name)
        expected_type = type_hints[field_name]

        if not isinstance(field_value, expected_type):
            raise TypeError(f"Field '{field_name}' expected {expected_type}, got {type(field_value)}.")


def calculate_fragment_duration_s(batch_sizes, n_runners, step_duration_s):
    max_fragment_length = max(batch_sizes) / max(1, n_runners)
    step_duration_s = step_duration_s
    fragment_duration_s = max_fragment_length * step_duration_s

    print(
        f"Estimated time for collecting samples: {fragment_duration_s:.1f}s"
        " (max_batch_size=%d, n_runners=%d, step_duration_s=%s)" % (
            max(batch_sizes), max(1, n_runners), step_duration_s
        )
    )

    # Maximum allowed time for sample collection (hard-coded)
    max_fragment_duration_s = 30

    if fragment_duration_s > max_fragment_duration_s:
        raise Exception(
            "Estimated fragment_duration_s is too big: %.1f (based on step_duration_s=%s).\n"
            "To fix this, either:\n"
            "\t* Increase train env runners (current: %d)\n"
            "\t* Decrease train_batch_size_per_learner (current: %s)\n"
            "\t* Increase max_fragment_duration_s (current: %d, hard-coded)" % (
                fragment_duration_s, step_duration_s,
                n_runners, batch_sizes,
                max_fragment_duration_s
            )
        )

    return fragment_duration_s


def calc_train_sample_timeout_s(batch_sizes, n_runners, step_duration_s, headroom=10):
    max_fragment_duration_s = calculate_fragment_duration_s(batch_sizes, n_runners, step_duration_s)
    sample_timeout_s = max_fragment_duration_s * headroom
    return sample_timeout_s


def calc_eval_sample_timeout_s(n_episodes, n_runners, step_duration_s, max_episode_len, headroom=10):
    max_episode_duration_s = max_episode_len * step_duration_s
    sample_timeout_s = max_episode_duration_s * headroom
    return sample_timeout_s


def get_env_cls(gym_env_name):
    env_spec = gym.envs.registration.registry[gym_env_name]
    env_cls = gym.envs.registration.load_env_creator(env_spec.entry_point)
    return env_cls
