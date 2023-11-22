import collections
import enum
import numpy as np

from .pyconnector import ERRNAMES

# the numpy data type (pytorch works best with float32)
DTYPE = np.float32

ActionType = enum.IntEnum("Color", [
    "WAIT", "MOVE", "MOVE1",
    "MOVE2", "MOVE3", "MOVE4",
    "MOVE5", "MOVE6", "MOVE7",
])

Analysis = collections.namedtuple("Analysis", [
    "action_type",                      # int <ActionType>
    "action_type_counters_ep",          # int[len(ActionType)]
    "action_type_valid_counters_ep",    # int[len(ActionType)]
    "action_hex",                       # int
    "action_hex_counters_ep",           # int[len(15*11)]
    "action_hex_valid_counters_ep",     # int[len(15*11)]
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


class Analyzer():
    # NOTES on action_type and action_hex counters:
    # The idea is to visualise action_type as a heatmap where
    # x-axis is divided into N regions (time intervals)
    # y-axis is diveded into 9 regions (action types)
    #
    # For action_hex, the idea is a line chart with a heatmap popup:
    # x-axis is divided into N regions (time intervals)
    # y-axis is divided into 165 regions (battlefield hexes)
    # The line will show the AVG(value) of that hex at time T
    # The line chart itself is not informative, BUT:
    #
    # On hover, a popup will show a 2D heatmap where:
    # x-axis is divided into 15 regions (bfield hex witdth)
    # y-axis is diveded into 11 regions (bfield hex height)
    # The hexes will be colored according to their at time T

    def __init__(self, n_actions, errflags):
        self.errflags = errflags

        # def, wait, 165*8 moves
        exp_n_actions = 2 + 165*8
        assert n_actions == exp_n_actions, "Expected %d actions, got: %d" % (exp_n_actions, n_actions)

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
        self.action_hex_counters = np.zeros(15 * 11, dtype=DTYPE)
        self.action_hex_valid_counters = np.zeros(15 * 11, dtype=DTYPE)

    def analyze(self, act, res):
        self.actions_count += 1

        errors_count, error_counters = self._parse_errmask(res.errmask)
        action_type, action_hex = self._action_type_and_hex(act)
        self.action_type_counters[action_type] += 1
        self.action_hex_counters[action_hex] += 1

        if errors_count > 0:
            self.actions_valid_consecutive = 0
            self.errors_count += errors_count
            self.error_counters += error_counters
            self.errors_consecutive_count += 1
        else:
            self.errors_consecutive_count = 0
            self.actions_valid_count += 1
            self.actions_valid_consecutive += 1
            self.action_type_valid_counters[action_type] += 1
            self.action_hex_valid_counters[action_hex] += 1

        net_dmg = res.dmg_dealt - res.dmg_received
        self.net_dmg += net_dmg

        net_value = res.value_killed - res.value_lost
        self.net_value += net_value

        return Analyzer.Analysis(
            action_type=action_type,
            action_type_counters_ep=self.action_type_counters,
            action_type_valid_counters_ep=self.action_type_valid_counters,
            action_hex=action_hex,
            action_hex_counters_ep=self.action_hex_counters,
            action_hex_valid_counters_ep=self.action_hex_valid_counters,
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
    act => act0
    0 = defend                => DEFEND
    1 = wait                  => WAIT
    2 = move(0,0)             => MOVE
    3 = move(0,0)+Attack#1    => MOVE1
    ...
    9  = move(0,0)+Attack#7   => MOVE7
    10 = move(1,0)            => MOVE
    ...
    1322 = move(15,11)+Attack#7 => MOVE7
    """
    def _action_type_and_hex(self, act):
        if act < 2:
            return ActionType(act), None
        else:
            act0 = act - 2
            return ActionType(act0 % 8), act0 / 8

    def _update_metrics(self, analysis):
        if analysis.n_errors:
            assert analysis.net_value == 0
            self.errcounters += analysis.errcounters
            self.n_errors_consecutive += analysis.n_errors
        else:
            self.net_value += analysis.net_value
            self.n_steps_successful += 1
            self.n_errors_consecutive = 0
