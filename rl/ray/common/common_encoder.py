from dataclasses import dataclass, field
import torch
import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, ACTOR, CRITIC


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


class Common_Encoder(torch.nn.Module):
    def __init__(
        self,
        shared,
        inference_only,
        action_space,
        observation_space,
        obs_dims,
        netconfig,
    ):
        super().__init__()

        assert isinstance(obs_dims, dict)
        assert list(obs_dims.keys()) == ["misc", "stacks", "hexes"]  # order is important

        self.shared = shared
        self.inference_only = inference_only
        self.encoder = EncoderNN(netconfig, action_space, observation_space, obs_dims)

        # XXX: The "critic_encoder" variable is needed by PPORLModule
        #   Using the same var here means critic encoder will be
        #   simply deleted by PPORLModule if not needed (e.g. during inference)
        #
        if shared:
            pass  # critic_encoder must be UNDEFINED if shared is True
        else:
            self.critic_encoder = EncoderNN(netconfig, action_space, observation_space, obs_dims)

    def forward(self, batch) -> torch.Tensor:
        # LSTM-only:
        # state_actor = batch[Columns.STATE_IN][ACTOR]
        # state_critic = batch[Columns.STATE_IN][CRITIC]
        obs = batch[Columns.OBS]["observation"]
        out = {ACTOR: self.encoder(obs)}

        if not self.inference_only:
            out[CRITIC] = out[ACTOR] if self.shared else self.critic_encoder(obs)

        # LSTM-only:
        # out[Columns.STATE_OUT][ACTOR] = {"h": ..., "c": ...}
        # out[Columns.STATE_OUT][CRITIC] = {"h": ..., "c": ...}

        return {ENCODER_OUT: out}

    # XXX: custom encoder call by the RLModule's overriden compute_values()
    def compute_value(self, batch) -> torch.Tensor:
        # RLModule should directly use out[CRITIC] instead
        assert not self.shared, "compute_values() should not be called when critic is shared"
        assert not self.inference_only
        assert hasattr(self, "critic_encoder")

        # LSTM-only:
        # state_actor = batch[Columns.STATE_IN][ACTOR]
        # state_critic = batch[Columns.STATE_IN][CRITIC]
        obs = batch[Columns.OBS]["observation"]

        # XXX: compute_values is called during training (no output states needed)
        #      The output is also a plain tensor as it's simply fed to the VF
        return self.critic_encoder(obs)


class EncoderNN(torch.nn.Module):
    def __init__(self, netconfig, action_space, observation_space, obs_dims):
        super().__init__()

        self.obs_splitter = Split(list(obs_dims.values()), dim=1)

        self.features_extractor1_misc = torch.nn.Sequential()
        for spec in netconfig.features_extractor1_misc:
            layer = build_layer(spec, obs_dims)
            self.features_extractor1_misc.append(layer)

        dummy_outputs = []

        # dummy input to initialize lazy modules
        with torch.no_grad():
            dummy_outputs = [self.features_extractor1_misc(torch.randn([1, obs_dims["misc"]]))]

        for layer in self.features_extractor1_misc:
            layer_init(layer)

        self.features_extractor1_stacks = torch.nn.Sequential(
            torch.nn.Unflatten(dim=1, unflattened_size=[20, obs_dims["stacks"] // 20])
        )

        for spec in netconfig.features_extractor1_stacks:
            layer = build_layer(spec, obs_dims)
            self.features_extractor1_stacks.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            dummy_outputs.append(self.features_extractor1_stacks(torch.randn([1, obs_dims["stacks"]])))

        for layer in self.features_extractor1_stacks:
            layer_init(layer)

        self.features_extractor1_hexes = torch.nn.Sequential(
            torch.nn.Unflatten(dim=1, unflattened_size=[165, obs_dims["hexes"] // 165])
        )

        for spec in netconfig.features_extractor1_hexes:
            layer = build_layer(spec, obs_dims)
            self.features_extractor1_hexes.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            dummy_outputs.append(self.features_extractor1_hexes(torch.randn([1, obs_dims["hexes"]])))

        for layer in self.features_extractor1_hexes:
            layer_init(layer)

        self.features_extractor2 = torch.nn.Sequential()
        for spec in netconfig.features_extractor2:
            layer = build_layer(spec, obs_dims)
            self.features_extractor2.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            self.features_extractor2(torch.cat(tuple(dummy_outputs), dim=1))

        for layer in self.features_extractor2:
            layer_init(layer)

    def forward(self, x) -> torch.Tensor:
        misc, stacks, hexes = self.obs_splitter(x)
        fmisc = self.features_extractor1_misc(misc)
        fstacks = self.features_extractor1_stacks(stacks)
        fhexes = self.features_extractor1_hexes(hexes)
        features1 = torch.cat((fmisc, fstacks, fhexes), dim=1)
        return self.features_extractor2(features1)
