import copy
import random
import string


# "n" exponentially distributed numbers in the range [low, high]
def explist(low, high, n=100, dtype=float):
    x = (high/low) ** (1 / (n-1))
    return list(map(lambda i: dtype(low * x**i), range(n)))


# "n" linearly distributed numbers in the range [low, high]
def linlist(low, high, n=100, dtype=float):
    x = (high-low) / (n-1)
    return list(map(lambda i: dtype(low + x*i), range(n)))


def deepmerge(a: dict, b: dict, allow_new=True, update_existing=True, path=[]):
    if len(path) == 0:
        a = copy.deepcopy(a)

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


def get_divisors(n, nmin=0):
    divisors = [i for i in range(n // 2, nmin, -1) if n % i == 0]
    return divisors


def gen_id(n=8):
    population = string.ascii_lowercase + string.digits
    return str.join("", random.choices(population, k=n))
