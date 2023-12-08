import collections
import enum
import numpy as np

from .pyconnector import (
    N_ACTIONS,
    N_NONHEX_ACTIONS,
    N_HEX_ACTIONS,
)

# the numpy data type (pytorch works best with float32)
DTYPE = np.float32

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

    RETREAT = 0
    WAIT = enum.auto()
    assert N_NONHEX_ACTIONS == 2

    MOVE = enum.auto()
    SHOOT = enum.auto()
    MELEE_TL = enum.auto()
    MELEE_TR = enum.auto()
    MELEE_R = enum.auto()
    MELEE_BR = enum.auto()
    MELEE_BL = enum.auto()
    MELEE_L = enum.auto()
    MELEE_T = enum.auto()
    MELEE_B = enum.auto()
    assert N_HEX_ACTIONS == 10


class Analyzer():
    def __init__(self, action_offset):
        self.actions_count = 0
        self.net_dmg = 0
        self.net_value = 0
        self.action_type_counters = np.zeros(len(ActionType), dtype=DTYPE)

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
    2 = MOVE(0,0)
    3 = SHOOT(0,0)
    4 = MELEE_TL(0,0)
    5 = MELEE_TR(0,0)
    ...
    12 = MOVE(1,0)
    13 = SHOOT(1,0)
    14 = MELEE_TL(1,0)
    ...
    1651 = MELEE_B(15,11)
    """
    def _action_type(self, act):
        if act < N_NONHEX_ACTIONS:
            return ActionType(act)
        else:
            hexact = act - N_NONHEX_ACTIONS
            return ActionType(N_NONHEX_ACTIONS + hexact % N_HEX_ACTIONS)
