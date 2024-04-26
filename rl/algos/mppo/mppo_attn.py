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
    STATE_VALUE_NA,
    Simo
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
    """
    def _qk_mask(self, q: Hex, k: Hex):
        if q.HEX_STATE != 1:  # != OCCUPIED
            return 0

        score = 0

        q_is_active = bool(q.STACK_IS_ACTIVE)
        k_is_active = bool(k.STACK_IS_ACTIVE)
        q_is_shooter = bool(q.STACK_SHOTS)
        k_is_shooter = bool(k.STACK_SHOTS)
        q_is_right = q.STACK_SIDE == 1
        k_is_right = q.STACK_SIDE == 1
        assert q.STACK_SIDE is not None  # would not be here if q is blank
        k_is_blank = k.STACK_SIDE is None

        # None means no such stack
        attacker_stacks = [i for i in range(7) if q.getattr("HEX_REACHABLE_BY_FRIENDLY_STACK_{i}") is not None]
        defender_stacks = [i for i in range(7) if q.getattr("HEX_REACHABLE_BY_ENEMY_STACK_{i}") is not None]

        # TODO: sanity asserts

        q_slot = q.STACK_SLOT

        if q.STACK_IS_ACTIVE:
            # boost q's attention at k (q is the active stack)
            mul = 5 if q.STACK_IS_WIDE else 10

            tmpmul = 2 if k_is_defender \
                else 1 if k_is_blank \
                else 0 # if k_is_attacker
            score += tmpmul * mul * k.HEX_SHOOTABLE_BY_ACTIVE_STACK

            if k.HEX_REACHABLE_BY_ACTIVE_STACK:
                tmpmul = 0
                for i in defender_stacks:
                    tmpmul += 1 if 
















        if q_is_active:
            def k_hexattr_wrt_q(name):
                return 2 * getattr(k, f"HEX_{name}_ACTIVE_STACK")
        elif q_is_friendly:
            def k_hexattr_wrt_q(name):
                return getattr(k, f"HEX_{name}_FRIENDLY_STACK_{qSlot}")
        else:
            def k_hexattr_wrt_q(name):
                return getattr(k, f"HEX_{name}_ENEMY_STACK_{qSlot}")

        if q_is_enemy:
            score += 2 * k_hexattr_wrt_q("SHOOTABLE_BY")
        else:
            score += k_hexattr_wrt_q("SHOOTABLE_BY")

        # "I can move to k" (0..1)
        score += k_hexattr_wrt_q("REACHABLE_BY")


        if k_hexattr_wrt_q("REACHABLE_BY"):

            for i in range(7):
                # "... and I will be next to an enemy"
                if getattr(k, "HEX_NEXT_TO_ENEMY_STACK_{i}"):

            score += k_hexattr_wrt_q("REACHABLE_BY")

        if k.HEX_NEXT_TO_ENEMY_STACK_0:
            mult = mult0 + 1 if 


            # as a shooter, pay attention to enemy targets:

            if k.HEX_NEXT_TO_ENEMY_STACK and k.HEX_REACHABLE_BY_ACTIVE_STACK:


            score += mult * k.HEX_REACHABLE_BY_ACTIVE_STACK  # 0..1


            score += mult * k.HEX_MELEEABLE_BY_ACTIVE_STACK  # 0..2
            score += mult * k.HEX_SHOOTABLE_BY_ACTIVE_STACK  # 0..2
            # score += mult * k.NEXT_TO_ACTIVE_STACK  # 0..1



                score += 1

                if k.STACK_SHOTS
                if k.STACK_SIDE == self.otherside:  # enemy blocks active shooter
                    score += 1
                elif k.STACK_SIDE == self.side:  # friendly protects active shooter
                    # active stack shooting is blocked by enemy
                    score += 1

                if k.HEX_NEXT_TO_ACTIVE_STACK:
                    if q.STACK_SHOTS or 

                score += mult * k.HEX_NEXT_TO_ACTIVE_STACK  # 0..1


        if q.STACK_SIDE == self.side:
            score += getattr(k, f"HEX_REACHABLE_BY_FRIENDLY_STACK_{q.STACK_SLOT}")
            score += getattr(k, f"HEX_MELEEABLE_BY_FRIENDLY_STACK_{q.STACK_SLOT}")
            score += getattr(k, f"HEX_SHOOTABLE_BY_FRIENDLY_STACK_{q.STACK_SLOT}")
            score += getattr(k, f"HEX_NEXT_TO_FRIENDLY_STACK_{q.STACK_SLOT}")
        elif q.STACK_SIDE == self.otherside:
            score += getattr(k, f"HEX_REACHABLE_BY_ENEMY_STACK_{q.STACK_SLOT}")
            score += getattr(k, f"HEX_MELEEABLE_BY_ENEMY_STACK_{q.STACK_SLOT}")
            score += getattr(k, f"HEX_SHOOTABLE_BY_ENEMY_STACK_{q.STACK_SLOT}")
            score += getattr(k, f"HEX_NEXT_TO_ENEMY_STACK_{q.STACK_SLOT}")
        else:  # no stack





        if q.STACK_IS_ACTIVE:
            if k.HEX_STATE == 0: return True  # k is OBSTACLE
            if k.IS_ACTIVE: return True
            if k.STACK_SIDE == self.side: return True




class Agent(mppo.Agent):
    def __init__(self, args, observation_space, action_space, state=None):
        super().__init__()
        self.NN = mppo.AgentNN(args.network, action_space, observation_space)


def main(args):
    mppo.main(args, agent_cls=Agent)


if __name__ == "__main__":
    main(mppo.debug_args())
