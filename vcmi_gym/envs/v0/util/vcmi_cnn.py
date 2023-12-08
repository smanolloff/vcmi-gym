import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn


class VcmiCNN(BaseFeaturesExtractor):
    # x1 = hex 1
    # n = hexstate + n-1 stack attributes

    # Observation (11, 15, n):
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
    # We can only represent a grid where each hex is n consecutive elements:
    #  __hex__  __hex__  __hex__     __hex__
    #  x x...x  x x...x  x x...x ... x x...x
    #  ...
    #
    # We flatten it to a (11, 15*n) (the size of 1hex is (1,n) aka. (n,))
    # The kernel is (3, 3*n) (ie. 3x3 hexes).
    # The stride is (1, n) (ie. 1 hex)
    # The padding is (1, n) (ie. 1 hex each side).
    #   NOTE: padding_mode=zeros assumes a "0" hexstate is an INVALID hex!

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,  # XXX: no idea how many "features" to set...
    ) -> None:
        assert isinstance(observation_space, gym.spaces.Box)
        super().__init__(observation_space, features_dim)

        assert len(observation_space.shape) == 2, "unexpected shape"
        assert observation_space.shape[0] == 11, "unexpected height"
        assert observation_space.shape[1] % 15 == 0, "unexpected width"

        n = observation_space.shape[1] // 15

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, n), stride=(1, n), padding=0),
            nn.ReLU(),
            # => (11, 15) matrix
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # => (11, 15) matrix
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
