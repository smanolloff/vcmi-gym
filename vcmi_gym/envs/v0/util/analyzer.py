import collections
import enum
import numpy as np

from .pyconnector import ERRNAMES, N_ACTIONS

# the numpy data type (pytorch works best with float32)
DTYPE = np.float32

Analysis = collections.namedtuple("Analysis", [
    "action_type",                      # int <ActionType>
    "action_type_counters_ep",          # int[len(ActionType)]
    "action_type_valid_counters_ep",    # int[len(ActionType)]
    "action_coords",                    # (int, int) or None
    "action_coords_counters_ep",        # int[11][15]
    "action_coords_valid_counters_ep",  # int[11][15]
    "actions_count_ep",                 # int
    "actions_valid_count_ep",           # int
    "actions_valid_consecutive",        # int
    "errors_count",                     # int
    "errors_count_ep",                  # int
    "errors_consecutive_count",         # int
    "error_counters",                   # int[len(ERRNAMES))
    "error_counters_ep",                # int[len(ERRNAMES))
    "net_dmg",                          # int
    "net_dmg_ep",                       # int
    "net_value",                        # int
    "net_value_ep",                     # int
])


class ActionType(enum.IntEnum):
    # 3 non-move actions
    RETREAT = 0
    DEFEND = enum.auto()
    WAIT = enum.auto()

    # 8 move actions (associated with 1 of 165 hexes)
    MOVE = enum.auto()
    MOVE1 = enum.auto()
    MOVE2 = enum.auto()
    MOVE3 = enum.auto()
    MOVE4 = enum.auto()
    MOVE5 = enum.auto()
    MOVE6 = enum.auto()
    MOVE7 = enum.auto()

    assert N_ACTIONS == 3 + 165*8


class Analyzer():
    # NOTES on action_type and action_coords_counters counters:
    #
    # The idea is to visualise action_type as a heatmap where
    # x-axis is divided into N regions (time intervals)
    # y-axis is diveded into 9 regions (action types)
    # Colors coresponding to action_types' values at time T
    #
    # For action_coords_counters, the idea is a line chart with a heatmap popup:
    # x-axis is divided into N regions (time intervals)
    # y-axis is 0..1
    # The line will show the mode of the distribution of actions across hexes at time T
    # That line would indicate intervals at which single hexes have been heavily preferred
    # An advanced approach would be to measure the distribution across clusters of hexes
    # On hover, a popup will show a 2D heatmap where:
    # x-axis is divided into 15 regions (bfield hex witdth)
    # y-axis is diveded into 11 regions (bfield hex height)
    # The hexes will be colored according to their value at time T

    def __init__(self, action_offset, errflags):
        self.errflags = errflags

        self.actions_count = 0
        self.actions_valid_count = 0
        self.actions_valid_consecutive = 0
        self.errors_count = 0
        self.errors_consecutive_count = 0
        self.error_counters = np.zeros(len(ERRNAMES), dtype=DTYPE)
        self.net_dmg = 0
        self.net_value = 0
        self.action_type_counters = np.zeros(len(ActionType), dtype=DTYPE)
        self.action_type_valid_counters = np.zeros(len(ActionType), dtype=DTYPE)
        self.action_coords_counters = np.zeros((11, 15), dtype=DTYPE)
        self.action_coords_valid_counters = np.zeros((11, 15), dtype=DTYPE)

    def analyze(self, act, res):
        self.actions_count += 1

        errors_count, error_counters = self._parse_errmask(res.errmask)
        action_type, coords = self._action_type_and_coords(act)
        self.action_type_counters[action_type] += 1

        if errors_count == 0:
            self.errors_consecutive_count = 0
            self.actions_valid_count += 1
            self.actions_valid_consecutive += 1
            self.action_type_valid_counters[action_type] += 1
        else:
            self.actions_valid_consecutive = 0
            self.errors_count += errors_count
            self.error_counters += error_counters
            self.errors_consecutive_count += 1

        if coords:
            x, y = coords
            self.action_coords_counters[y][x] += 1
            if errors_count == 0:
                self.action_coords_valid_counters[y][x] += 1

        net_dmg = res.dmg_dealt - res.dmg_received
        self.net_dmg += net_dmg

        net_value = res.value_killed - res.value_lost
        self.net_value += net_value

        return Analysis(
            action_type=action_type,
            action_type_counters_ep=self.action_type_counters,
            action_type_valid_counters_ep=self.action_type_valid_counters,
            action_coords=coords,
            action_coords_counters_ep=self.action_coords_counters,
            action_coords_valid_counters_ep=self.action_coords_valid_counters,
            actions_count_ep=self.actions_count,
            actions_valid_count_ep=self.actions_valid_count,
            actions_valid_consecutive=self.actions_valid_consecutive,
            errors_count=errors_count,
            errors_count_ep=self.errors_count,
            errors_consecutive_count=self.errors_consecutive_count,
            error_counters=error_counters,
            error_counters_ep=self.error_counters,
            net_dmg=net_dmg,
            net_dmg_ep=self.net_dmg,
            net_value=net_value,
            net_value_ep=self.net_value,
        )

    #
    # private
    #

    def _parse_errmask(self, errmask):
        n_errors = 0
        errcounters = np.zeros(len(self.errflags), dtype=DTYPE)

        for i, flag in enumerate(self.errflags):
            if errmask & flag:
                errcounters[i] += 1
                n_errors += 1

        return n_errors, errcounters

    """
    act => ActionType
    0 = retreat               => RETREAT
    1 = defend                => DEFEND
    2 = wait                  => WAIT
    3 = move(0,0)             => MOVE
    4 = move(0,0)+Attack#1    => MOVE1
    ...
    10 = move(0,0)+Attack#7   => MOVE7
    11 = move(1,0)            => MOVE
    12 = move(1,0)+Attack#1   => MOVE1
    ...
    1322 = move(15,11)+Attack#7 => MOVE7
    """
    def _action_type_and_coords(self, act):
        if act < 3:
            return ActionType(act), None
        else:
            hexact = act - 3
            hexnum = hexact // 8
            hexcoords = (hexnum % 15, hexnum // 15)
            return ActionType(3 + hexact % 8), hexcoords

    def _update_metrics(self, analysis):
        if analysis.n_errors:
            assert analysis.net_value == 0
            self.errcounters += analysis.errcounters
            self.n_errors_consecutive += analysis.n_errors
        else:
            self.net_value += analysis.net_value
            self.n_steps_successful += 1
            self.n_errors_consecutive = 0
