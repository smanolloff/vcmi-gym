import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn

from .pyconnector import (
    STATE_SIZE_X,
    STATE_SIZE_Y,
    STATE_SIZE_Z,
    N_HEX_ATTRS
)


class VcmiNN(BaseFeaturesExtractor):
    # x1 = hex 1
    # n = hexstate + n-1 stack attributes

    # Observation is a 2D array of shape (11, 15*n):
    # [
    #   [x1_1,   ...   x1_n, | x2_1,   ...   x2_n, | ... | x15_1,  ...  x15_n],
    #   [x16_1,  ...  x16_n, | x17_1,  ...  x17_n, | ... | x30_1,  ...  x30_n],
    #            ...                   ...                         ...
    #   [x151_1, ... x151_n, | x152_1, ... x152_n, | ... | x165_1, ... x165_n]
    # ]                      ^
    #                        logical hex separator (for vis. purposes)
    #
    # This corresponds to the battlefield:
    #
    #   x x x ... x
    #  x x x x ... x
    #   x x x ... x
    #  ...
    #
    # Essentially a grid where each hex is n consecutive elements:
    #  __hex__  __hex__  __hex__     __hex__
    #  x x...x  x x...x  x x...x ... x x...x
    #  ...
    #
    # Example config
    #       VcmiNN(
    #           layers=[
    #               {"t": "Conv2d", "out_channels": 32, "kernel_size": (1, 15), "stride": (1, 15), "padding": 0},
    #               {"t": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
    #               {"t": "Conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2},
    #           ],
    #           activation="ReLU",
    #           output_dim=1024,
    #       )
    def __init__(
        self,
        observation_space: gym.Space,
        layers,
        activation,
        output_dim
    ) -> None:
        assert isinstance(observation_space, gym.spaces.Box)
        super().__init__(observation_space, output_dim)

        # It does not work if we don't have an explicit "channel" dim in obs
        assert len(observation_space.shape) == 3, "unexpected shape"

        # The "Z" is apparently required, as SB3 uses it for passing multiple
        # observations at once:
        #   Z=batch_size during train()
        #   Z=n_envs during collect_rollouts()
        # I may be wrong here, but without it, everything got messed up
        assert observation_space.shape[0] == STATE_SIZE_Z, "expected channels=%d" % STATE_SIZE_Z

        # Y=11 (battlefield height)
        assert observation_space.shape[1] == STATE_SIZE_Y, "expected height=%d" % STATE_SIZE_Y
        # X=15*15 (battlefield width is 15 hexes, each with 15 attrs)
        # ideally, X=15 and Z=15, but handling Z is was too difficult in pybind
        # so we use X=15*15 and Z=1
        assert observation_space.shape[2] == STATE_SIZE_X, "expected width=%d" % STATE_SIZE_X
        assert observation_space.shape[2] % N_HEX_ATTRS == 0, "width to be divisible by %d" % N_HEX_ATTRS

        activation_cls = getattr(nn, activation)

        self.network = nn.Sequential()
        for (i, layer) in enumerate(layers):
            # fallback to Conv2d for old .zip models ("_" key not stored)
            layer_cls = getattr(nn, layer.pop("t", "Conv2d"))
            layer_kwargs = dict(layer)  # copy

            if i == 0:
                assert "in_channels" not in layer_kwargs, "in_channels must be omitted for 1st layer"
                layer_kwargs["in_channels"] = STATE_SIZE_Z

            self.network.append(layer_cls(**layer_kwargs))
            self.network.append(activation_cls())

        # Always flatten to "out_dim"
        self.network.append(nn.Flatten())

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.network(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.network.append(nn.Linear(n_flatten, output_dim))
        self.network.append(activation_cls())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(observations)
