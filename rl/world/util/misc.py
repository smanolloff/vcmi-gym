import numpy as np
import torch


def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)
    for mod in list(layer.modules())[1:]:
        layer_init(mod, gain, bias_const)
    return layer


def dig(data, *keys):
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data


def aggregate_metrics(queue):
    total = 0
    count = 0

    while not queue.empty():
        item = queue.get()
        total += item
        count += 1

    return total / count if count else None


def timer_stats(timers):
    res = {}
    t_all = timers["all"].peek()
    for k, v in timers.items():
        res[f"timer/{k}"] = v.peek()
        if k != "all":
            res[f"timer_rel/{k}"] = v.peek() / t_all

    res["timer/other"] = t_all - sum(v.peek() for k, v in timers.items() if k != "all")
    res["timer_rel/other"] = res["timer/other"] / t_all
    return res
