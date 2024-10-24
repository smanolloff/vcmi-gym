import torch
from torch import nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, ACTOR, CRITIC

from . import netutil


class MPPO_Encoder(nn.Module):
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
        # XXX: critic_encoder must be UNDEFINED if shared is True
        if not shared:
            self.critic_encoder = EncoderNN(netconfig, action_space, observation_space, obs_dims)

    def forward(self, batch) -> torch.Tensor:
        # XXX: if using LSTM:
        # state = batch[Columns.STATE_IN][CRITIC]

        obs = batch[Columns.OBS]["observation"]
        out = {ACTOR: self.encoder(obs)}

        if not self.inference_only:
            out[CRITIC] = out[ACTOR] if self.shared else self.critic_encoder(obs)

        # XXX: if using LSTM:
        # out[Columns.STATE_OUT] = {"h": ..., "c": ...}

        return {ENCODER_OUT: out}


class EncoderNN(nn.Module):
    def __init__(self, netconfig, action_space, observation_space, obs_dims):
        super().__init__()

        self.obs_splitter = netutil.Split(list(obs_dims.values()), dim=1)

        self.features_extractor1_misc = torch.nn.Sequential()
        for spec in netconfig.features_extractor1_misc:
            layer = netutil.build_layer(spec, obs_dims)
            self.features_extractor1_misc.append(layer)

        dummy_outputs = []

        # dummy input to initialize lazy modules
        with torch.no_grad():
            dummy_outputs = [self.features_extractor1_misc(torch.randn([1, obs_dims["misc"]]))]

        for layer in self.features_extractor1_misc:
            netutil.layer_init(layer)

        self.features_extractor1_stacks = torch.nn.Sequential(
            torch.nn.Unflatten(dim=1, unflattened_size=[20, obs_dims["stacks"] // 20])
        )

        for spec in netconfig.features_extractor1_stacks:
            layer = netutil.build_layer(spec, obs_dims)
            self.features_extractor1_stacks.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            dummy_outputs.append(self.features_extractor1_stacks(torch.randn([1, obs_dims["stacks"]])))

        for layer in self.features_extractor1_stacks:
            netutil.layer_init(layer)

        self.features_extractor1_hexes = torch.nn.Sequential(
            torch.nn.Unflatten(dim=1, unflattened_size=[165, obs_dims["hexes"] // 165])
        )

        for spec in netconfig.features_extractor1_hexes:
            layer = netutil.build_layer(spec, obs_dims)
            self.features_extractor1_hexes.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            dummy_outputs.append(self.features_extractor1_hexes(torch.randn([1, obs_dims["hexes"]])))

        for layer in self.features_extractor1_hexes:
            netutil.layer_init(layer)

        self.features_extractor2 = torch.nn.Sequential()
        for spec in netconfig.features_extractor2:
            layer = netutil.build_layer(spec, obs_dims)
            self.features_extractor2.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            self.features_extractor2(torch.cat(tuple(dummy_outputs), dim=1))

        for layer in self.features_extractor2:
            netutil.layer_init(layer)

    def forward(self, x) -> torch.Tensor:
        misc, stacks, hexes = self.obs_splitter(x)
        fmisc = self.features_extractor1_misc(misc)
        fstacks = self.features_extractor1_stacks(stacks)
        fhexes = self.features_extractor1_hexes(hexes)
        features1 = torch.cat((fmisc, fstacks, fhexes), dim=1)
        return self.features_extractor2(features1)
