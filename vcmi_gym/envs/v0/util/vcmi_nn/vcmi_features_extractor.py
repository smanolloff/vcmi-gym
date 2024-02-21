# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn

from ..pyconnector import (
    # STATE_SIZE_X,
    STATE_SIZE_Y,
    # STATE_SIZE_Z,
    # N_HEX_ATTRS
)


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


#
# Reshape a batched VCMI observation so that
# each attribute type is in a separate feature plane (channel)
#
# i.e. (B, 1, 11, 15*N) => (B, N, 11, 15)
#
# Example:
#   input = (B, 1, 11, 15*N)
#   output = (B, N, 11, 15)
#
# To test:
#   B = 2  # batch dim
#   N = 3  # 3 attributes per hex
#   obs = torch.as_tensor(np.ndarray((B,1,11,15*N), dtype="int"))
#   for b in range(B):
#       for y in range(11):
#           for x in range(15):
#               for a in range(N):
#                   if b == 0:
#                       obs[b][0][y][x*N+a] = 100 * (y*15 + x) + a
#                   else:
#                       obs[b][0][y][x*N+a] = -100 * (y*15 + x) - a
#
#   v = VcmiHexAttrsAsChannels(3,11,15)
#   v(obs)
class VcmiHexAttrsAsChannels(nn.Module):
    def __init__(self, n, y, x):
        self.n = n
        self.y = y
        self.x = x
        self.xy = self.x * self.y
        super().__init__()
    def forward(self, x):
        b = x.shape[0]
        tmp = reshape_fortran(x.flatten(), [self.n, b * self.xy]).reshape(b * self.n, self.xy)
        return reshape_fortran(tmp, [b, self.n, self.xy]).reshape(b, self.n, self.y, self.x)


class VcmiAttention(nn.MultiheadAttention):
    def forward(self, obs):
        # TODO: attn_mask
        res, _weights = super().forward(obs, obs, obs, need_weights=False, attn_mask=None)
        return res


class BatchReshape(nn.Module):
    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def forward(self, x):
        # XXX: view won't work sometimes, for some reason
        # return x.view(x.shape[0], *self.shape)

        return x.reshape(x.shape[0], *self.shape)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        self.dim0 = dim0
        self.dim1 = dim1
        super().__init__()

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class VcmiFeaturesExtractor(BaseFeaturesExtractor):
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
    #               {"t": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": (1, 15), "stride": (1, 15)},
    #               {"t": "LeakyReLU"},
    #               {"t": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
    #               {"t": "LeakyReLU"},
    #               {"t": "Conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2},
    #               {"t": "LeakyReLU"},
    #               {"t": "Flatten"},
    #               {"t": "Linear", "in_features": 384, "out_features": 1024},
    #               {"t": "LeakyReLU"}
    #           ],
    #           attention_kwargs=[
    #               layers=[...]
    #           ]
    #       )
    def __init__(self, observation_space: gym.Space, layers, attention_kwargs=None) -> None:
        assert isinstance(observation_space, gym.spaces.Box)

        # Conv2d does not work if we don't have an explicit "channel" dim in obs
        assert len(observation_space.shape) == 3, "unexpected shape"

        # The "Z" (ie. "channel") dim is always required:
        # conv2d is designed to process 2D images with Z channels, ie. 3D inputs
        # => Input to conv2d cannot be (11, 225), must be 3D: (1, 11, 225)
        #    When using VecFrameStack, Z=n_stack, eg. for 4: (4, 11, 225)
        #
        # HOWEVER, conv2d expects "3D (unbatched) or 4D (batched) input",
        # which means three is special handling for batches of images:
        # ie. (B, 1, 11, 225). In SB3, B is:
        #   * batch_size during train(), eg. 32
        #   * n_envs during collect_rollouts(), eg. 4
        #

        # Y=11 (battlefield height)
        assert observation_space.shape[1] == STATE_SIZE_Y, "expected height=%d for shape: %s" % (STATE_SIZE_Y, observation_space.shape)  # noqa: E501
        # X=15*15=225 (battlefield width is 15 hexes, each with 15 attrs)
        # ideally, Y=11, X=15, Z=15, but handling Z is too difficult in pybind
        # so we use X=15*15 and Z=1
        # XXX: using Z != 1 would also cause issues (see above notes for Z & B)
        # assert observation_space.shape[2] == STATE_SIZE_X, "expected width=%d for shape: %s" % (STATE_SIZE_X, observation_space.shape)  # noqa: E501
        # assert observation_space.shape[2] % N_HEX_ATTRS == 0, "width to be divisible by %d" % N_HEX_ATTRS

        network = nn.Sequential()

        for (i, layer) in enumerate(layers):
            layer_kwargs = dict(layer)  # copy
            t = layer_kwargs.pop("t")
            layer_cls = getattr(nn, t, None) or globals()[t]

            if i == 0 and t == "Conv2d" and "in_channels" not in layer_kwargs:
                layer_kwargs["in_channels"] = observation_space.shape[0]

            network.append(layer_cls(**layer_kwargs))

        # Compute output dim by doing one forward pass
        with th.no_grad():
            # ignore the batch dim
            out_dim = network(th.as_tensor(observation_space.sample()[None]).float()).shape[1:]

        assert len(out_dim) == 1, "expected a flattened 1-dim output shape, got: %s" % out_dim
        super().__init__(observation_space, out_dim[0])
        self.network = network

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(observations)
