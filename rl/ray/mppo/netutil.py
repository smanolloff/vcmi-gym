import torch
import numpy as np

from dataclasses import dataclass, field


@dataclass
class NetConfig:
    attention: dict = None
    features_extractor1_misc: list[dict] = field(default_factory=list)
    features_extractor1_stacks: list[dict] = field(default_factory=list)
    features_extractor1_hexes: list[dict] = field(default_factory=list)
    features_extractor2: list[dict] = field(default_factory=list)
    actor: dict = field(default_factory=dict)
    critic: dict = field(default_factory=dict)


class Split(torch.nn.Module):
    def __init__(self, split_size, dim):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.split_size, self.dim)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, split_size={self.split_size})"


def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    initializable_layers = (
        torch.nn.Linear,
        torch.nn.Conv2d,
        # TODO: other layers? Conv1d?
    )

    if isinstance(layer, initializable_layers):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)

    for mod in list(layer.modules())[1:]:
        layer_init(mod, gain, bias_const)

    return layer


def build_layer(spec, obs_dims):
    kwargs = dict(spec)  # copy
    t = kwargs.pop("t")

    assert len(obs_dims) == 3  # [M, S, H]

    for k, v in kwargs.items():
        if v == "_M_":
            kwargs[k] = obs_dims["misc"]
        if v == "_H_":
            assert obs_dims["hexes"] % 165 == 0
            kwargs[k] = obs_dims["hexes"] // 165
        if v == "_S_":
            assert obs_dims["stacks"] % 20 == 0
            kwargs[k] = obs_dims["stacks"] // 20

    layer_cls = getattr(torch.nn, t, None) or globals()[t]
    return layer_cls(**kwargs)
