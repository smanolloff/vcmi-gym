import numpy as np
import torch


# Pandas dataframe columns for losses to aggregate before logging as W&B tables
class TableColumn:
    STEP = "step"               # wandb.run.step
    STAGE = "stage"             # "train" / "test"
    ATTRIBUTE = "attribute"     # "BATTLE_SIDE" / ...
    CONTEXT = "context"         # "global" / "player" / "hex"
    DATATYPE = "datatype"       # "cont_abs" / "cont_rel" / "categorical" / ...
    LOSS = "loss"               # float

    @classmethod
    def as_list(cls):
        return [v for k, v in vars(cls).items() if k.isupper() and isinstance(v, str)]


def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    layer._layer_initialized = True
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)
    for mod in list(layer.modules())[1:]:
        if getattr(mod, "_layer_initialized", False):
            # This module will take care of its own initialization
            continue
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
        # res[f"timer/{k}"] = v.peek()
        if k != "all":
            res[f"timer_rel/{k}"] = v.peek() / t_all

    t_other = t_all - sum(v.peek() for k, v in timers.items() if k != "all")
    res["timer_rel/other"] = t_other / t_all
    res["timer_abs/all"] = t_all
    return res


def safe_mean(array_like) -> float:
    return np.nan if len(array_like) == 0 else float(np.mean(array_like))
