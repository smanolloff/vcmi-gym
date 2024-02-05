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

import numpy as np
import gymnasium as gym

from .vcmi_env import VcmiEnv, tracelog
from .util.pyconnector import N_NONHEX_ACTIONS, N_HEX_ACTIONS

# the numpy data type (pytorch works best with float32)
DTYPE = np.float32
ZERO = DTYPE(0)
ONE = DTYPE(1)

MAXLEN = 80

ARMY_VALUE_REF = 600_000


class VcmiMDEnv(VcmiEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.MultiDiscrete([
            N_NONHEX_ACTIONS + 165 - self.action_offset,
            N_HEX_ACTIONS
        ])

    def _transform_action(self, action):
        offset = N_NONHEX_ACTIONS - self.action_offset

        if action[0] < offset:
            res = action[0]
        else:
            res = offset + (action[0] - offset) * N_HEX_ACTIONS + action[1]

        res += self.action_offset
        # print("transformed action: %d" % res)
        return res

    @tracelog
    def action_masks(self):
        # Result is a [[HEX0, [ACT0, ACT1, ...ACT9]], [HEX1, [ACT0, ...]]]
        hexmasks = self.result.actmask[N_NONHEX_ACTIONS:].reshape((165, N_HEX_ACTIONS))
        nonhexmasks = np.zeros((N_NONHEX_ACTIONS, N_HEX_ACTIONS), dtype=bool)
        for i in range(N_NONHEX_ACTIONS):
            nonhexmasks[i][0] = self.result.actmask[i]

        masks = np.concatenate((nonhexmasks[self.action_offset:], hexmasks), axis=0)

        # sanity check
        assert self.action_offset == 1
        assert np.array_equal(np.concatenate((masks[0][0:1], masks[1:].flatten())), super().action_masks())

        return [[np.any(m), m] for m in masks]
