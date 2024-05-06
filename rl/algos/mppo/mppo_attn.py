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
# This file contains a modified version of CleanRL's PPO implementation:
# https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py

from . import mppo

from vcmi_gym.envs.v0.vcmi_env import (
    VcmiEnv,
    Hex,
    Action,
    State,
    DmgMod,
    ShootDistance,
    MeleeDistance,
    STATE_VALUE_NA
)

# import torch
from torch import nn
import numpy as np


class AgentNN(mppo.AgentNN):
    def __init__(self, network, action_space, observation_space):
        super().__init__(network, action_space, observation_space)
        # Assuming one-hot encoding (574 floats per hex)
        self.embed_dim = 574
        self.side = 0  # 0=attacker
        self.otherside = 1
        self.mha = nn.MultiheadAttention(embed_dim=574, num_heads=1, batch_first=True)

    def get_value(self, x, *args, **kwargs):
        return super().get_value(self._attention(x), *args, **kwargs)

    def get_action_and_value(self, x, *args, **kwargs):
        return super().get_action_and_value(self._attention(x), *args, **kwargs)

    def _attention(self, b_obs):
        assert b_obs.shape == (b_obs.shape[0], 11, 15, 574)

        b_obs = b_obs.flatten(start_dim=1, end_dim=2)
        # => (B, 165, 574)

        b_mask = np.zeros((b_obs.shape[0], 165, 165), dtype=bool)
        # => (B, 165, 165)

        for b, obs in enumerate(b_obs):
            # obs is (165, 86)
            decoded_obs = VcmiEnv.decode_obs(obs)

            for iq, q in enumerate(decoded_obs):
                for ik, k in enumerate(decoded_obs):
                    b_mask[iq][ik] = self._qk_mask(q, k)

        return self.mha(b_obs, b_obs, b_obs, attn_mask=b_mask, need_weights=False)

    """
    q is the decoded Hex we are standing at (the POV)
    k is the decoded Hex we are looking at

    We (as `q`) must determine how much attention to pay to `k`

    XXX: initial version returns a bool instead of a number
    """
    def _qk_mask(self, q: Hex, k: Hex):
        if q.HEX_STATE != State.OCCUPIED:
            return False

        if q.STACK_SIDE == Side.LEFT:
            return getattr(k, f"HEX_ACTION_MASK_FOR_L_STACK_{q.STACK_SLOT}") > 0
        else:
            return getattr(k, f"HEX_ACTION_MASK_FOR_R_STACK_{q.STACK_SLOT}") > 0


class Agent(mppo.Agent):
    def __init__(self, args, observation_space, action_space, state=None):
        super().__init__()
        self.NN = mppo.AgentNN(args.network, action_space, observation_space)


def main(args):
    mppo.main(args, agent_cls=Agent)


if __name__ == "__main__":
    main(mppo.debug_args())
