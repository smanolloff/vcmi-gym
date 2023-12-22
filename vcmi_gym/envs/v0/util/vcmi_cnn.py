import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn


class VcmiCNN(BaseFeaturesExtractor):
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
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,  # XXX: no idea how many "features" to set...
    ) -> None:
        assert isinstance(observation_space, gym.spaces.Box)
        super().__init__(observation_space, features_dim)

        # It does not work if we don't have an explicit "channel" dim in obs
        assert len(observation_space.shape) == 3, "unexpected shape"
        assert observation_space.shape[0] == 1, "expected channels=1"
        assert observation_space.shape[1] == 11, "expected height=11"
        assert observation_space.shape[2] % 15 == 0, "expected width divisible by 15"

        in_channels = observation_space.shape[0]
        n = observation_space.shape[2] // 15

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, n), stride=(1, n), padding=0),
            nn.ReLU(),
            # => (32, 11, 15)


            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # => (64, 11, 15)

            nn.Dropout(p=0.5),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # => (64, 11, 15)

            nn.Flatten(),
            # => (64, 165)
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
