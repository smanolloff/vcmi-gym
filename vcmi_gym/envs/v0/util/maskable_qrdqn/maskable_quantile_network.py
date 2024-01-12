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

from typing import Optional
import torch as th
import numpy as np
from sb3_contrib.qrdqn.policies import QuantileNetwork


class MaskableQuantileNetwork(QuantileNetwork):
    # obs is a tensor of size [N, obs_space.shape]
    # where N is: (TODO: confirm by debugging)
    #   - during collect_rollouts: N = n_envs
    #   - during train: N = batch_size
    #
    # quantiles is an array of size [N, n_quantiles, action_space.n], eg.:
    #
    #   [[[a1, a2, a3, ...an],  <- quantile 0 for all actions (N=0)
    #     [a1, a2, a3, ...an],  <- quantile 1 for all actions (N=0)
    #     [...]
    #     [a1, a2, a3, ...an]], <- quantile 200 for all actions (N=0)
    #    [[a1, a2, ...]]        <- quantile 0 for all actions (N=1)
    #     [...]]]
    #
    # What predict does is as follows:
    # quantiles.mean(dim=1)
    # => [[q0, q1, ... qn]  <- mean quantiles for each action (N=0)
    #     [q0, q1, ... q1]] <- mean quantiles for each action (N=1)
    #
    # then quantiles.argmax(dim=1)
    # => [5, 19]  <- quantile with index 5 for N=0 and index 19 for N=1
    # (ie. the actions)
    #
    # We must replace all quantiles for masked actions with -inf
    # Example:
    #
    #       import torch
    #       tensor = torch.tensor([[[1., 2.],
    #                               [3., 4.],
    #                               [5., 6.]],
    #                              [[7., 8.],
    #                               [9., 10.],
    #                               [11., 12.]],
    #                              [[13., 14.],
    #                               [15., 16.],
    #                               [17., 18.]]])
    #       mask = torch.tensor([[False, False, False],  # rows to keep in the first block
    #                            [False, True, True],   # rows to keep in the second block
    #                            [True, True, False]])  # rows to keep in the third block
    #       tensor[~mask] = 0  # Zero out values based on the mask
    #       print(tensor)
    def forward(self, obs: th.Tensor, action_masks: np.ndarray) -> th.Tensor:
        quantiles = self.quantile_net(self.extract_features(obs, self.features_extractor))
        view = quantiles.view(-1, self.n_quantiles, int(self.action_space.n))
        mask_false_indexes = th.nonzero(~th.as_tensor(action_masks)).flatten()
        view[:, :, mask_false_indexes] = th.finfo(quantiles.dtype).min
        return view

    def _predict(self, observation: th.Tensor, deterministic: bool = True, action_masks: Optional[np.ndarray] = None) -> th.Tensor:
        q_values = self(observation, action_masks).mean(dim=1)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action
