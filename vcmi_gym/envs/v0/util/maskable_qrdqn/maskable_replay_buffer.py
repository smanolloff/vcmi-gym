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

from typing import Optional, NamedTuple
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces
import numpy as np
import torch as th


class MaskableReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor


class MaskableReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_masks = None

        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.bool)

    def reset(self) -> None:
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.bool)
        super().reset()

    def add(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        assert action_masks is not None
        self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
        super().add(*args, **kwargs)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MaskableReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.action_masks[batch_inds, env_indices, :],
        )

        return MaskableReplayBufferSamples(*tuple(map(self.to_torch, data)))
