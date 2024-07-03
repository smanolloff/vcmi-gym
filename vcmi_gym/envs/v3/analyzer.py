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

import collections
import enum
import numpy as np

from .pyconnector import (
    N_ACTIONS,
    N_NONHEX_ACTIONS,
    N_HEX_ACTIONS,
)

Analysis = collections.namedtuple("Analysis", [
    "action_type",                      # int <ActionType>
    "action_type_counters_ep",          # int[len(ActionType)]
    "actions_count_ep",                 # int
    "net_dmg",                          # int
    "net_dmg_ep",                       # int
    "net_value",                        # int
    "net_value_ep",                     # int
])


class ActionType(enum.IntEnum):
    assert N_ACTIONS == N_NONHEX_ACTIONS + 165*N_HEX_ACTIONS
    assert N_NONHEX_ACTIONS == 2
    assert N_HEX_ACTIONS == 14

    # Nonhex-actions
    RETREAT = 0
    WAIT = enum.auto()

    # Hex-actions
    MOVE = enum.auto()
    AMOVE_TL = enum.auto()
    AMOVE_TR = enum.auto()
    AMOVE_R = enum.auto()
    AMOVE_BR = enum.auto()
    AMOVE_BL = enum.auto()
    AMOVE_L = enum.auto()
    AMOVE_2BL = enum.auto()
    AMOVE_2L = enum.auto()
    AMOVE_2TL = enum.auto()
    AMOVE_2TR = enum.auto()
    AMOVE_2R = enum.auto()
    AMOVE_2BR = enum.auto()
    SHOOT = enum.auto()


class Analyzer():
    def __init__(self, action_offset):
        self.actions_count = 0
        self.net_dmg = 0
        self.net_value = 0
        self.action_type_counters = np.zeros(len(ActionType), dtype=np.float32)

    def analyze(self, act, res):
        self.actions_count += 1

        action_type = self._action_type(act)
        self.action_type_counters[action_type] += 1

        net_dmg = res.dmg_dealt - res.dmg_received
        net_value = res.value_killed - res.value_lost
        self.net_dmg += net_dmg
        self.net_value += net_value

        return Analysis(
            action_type=action_type,
            action_type_counters_ep=self.action_type_counters,
            actions_count_ep=self.actions_count,
            net_dmg=net_dmg,
            net_dmg_ep=self.net_dmg,
            net_value=net_value,
            net_value_ep=self.net_value,
        )

    """
    act => ActionType
    0 = RETREAT
    1 = WAIT
    2 = AMOVE_TR(0,0)
    ...
    14 = MOVE(0,0)
    15 = SHOOT(0,0)
    16 = MELEE_TR(0,1)
    ...
    2311 = SHOOT(11,15)
    """
    def _action_type(self, act):
        if act < N_NONHEX_ACTIONS:
            return ActionType(act)
        else:
            hexact = act - N_NONHEX_ACTIONS
            return ActionType(N_NONHEX_ACTIONS + hexact % N_HEX_ACTIONS)
