import time
import numpy as np
import gymnasium as gym
import os

from .connector.build import connector
from .util import log

# the numpy data type (pytorch works best with float32)
DTYPE = np.float32
ZERO = DTYPE(0)
ONE = DTYPE(1)


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    VCMI_LOGLEVELS = ["trace", "debug", "info", "warn", "error"]
    ERROR_MAPPING = connector.get_error_mapping()
    INFO_KEYS = ["is_success", "errors"] + [name for (name, _) in ERROR_MAPPING.values()]

    def __init__(
        self,
        mapname,
        seed=None,  # not used currently
        render_mode="ansi",
        vcmi_loglevel_global="error",
        vcmi_loglevel_ai="error"
    ):
        assert vcmi_loglevel_global in VcmiEnv.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in VcmiEnv.VCMI_LOGLEVELS

        self.render_mode = render_mode
        self.errflags = VcmiEnv.ERROR_MAPPING.keys()
        self.errnames = [name for (name, desc) in VcmiEnv.ERROR_MAPPING.values()]
        self.errcounters = np.zeros(len(self.errnames), dtype=DTYPE)
        self.logger = log.get_logger("VcmiEnv", "DEBUG")

        # NOTE: removing action=0 (retreat) which is used for resetting...
        #       => start from 1 and reduce total actions by 1
        #          XXX: there seems to be a bug as start=1 causes this error:
        #          index 1322 is out of bounds for dimension 2 with size 1322
        #       => just start from 0, reduce max by 1, and manually add +1
        self.action_space = gym.spaces.Discrete(connector.get_n_actions() - 1)
        self.observation_space = gym.spaces.Box(shape=(connector.get_state_size(),), low=-1, high=1, dtype=DTYPE)
        self.connector = connector.Connector(mapname, vcmi_loglevel_global, vcmi_loglevel_ai)
        self.result = self.connector.start()
        self.n_steps = 0
        self.render_mode = "ansi"

    def step(self, action):
        if self.result.get_is_battle_over():
            raise Exception("Reset needed")

        self.n_steps += 1
        self.result = self.connector.act(action + 1)
        reward = self.calc_reward(self.result)
        return self.result.get_state(), reward, self.result.get_is_battle_over(), False, self.build_info()

    def reset(self, seed=None, options=None):
        if self.n_steps == 0:
            return self.result.get_state(), self.build_info()

        super().reset(seed=seed)

        self.errcounters = np.zeros(len(self.errnames), dtype=DTYPE)
        self.result = self.connector.reset()
        return self.result.get_state(), self.build_info()

    def render(self):
        if self.render_mode == "ansi":
            return self.connector.renderAnsi()
        elif self.render_mode == "rgb_array":
            gym.logger.warn("Rendering RGB arrays not yet implemented for VcmiEnv")
        elif self.render_mode is None:
            gym.logger.warn("Cannot render with no render mode set")

        return

    def close(self):
        # graceful shutdown of VCMI is not impossible (by design)
        # The 10+ running threads lead to segfaults on any such attempt anyway
        gym.logger.warn("Graceful shutdown not yet implemented for VcmiEnv")

    def error_summary(self):
        res = "Error summary:\n"
        for (i, name) in enumerate(self.errnames):
            res += ("%25s: %d\n" % (name, self.errcounters[i]))

        return res

    #
    # private
    #

    def calc_reward(self, res):
        self.last_action_n_errors = self.parse_errmask(res.get_errmask())
        # self.logger.debug("Action errors: %d" % n_errors)

        if self.last_action_n_errors:
            return -10 * self.last_action_n_errors

        net_dmg = res.get_dmg_dealt() - res.get_dmg_received()
        net_value = res.get_value_killed() - res.get_value_lost()

        return net_value + 10*net_dmg

    def parse_errmask(self, errmask):
        n_errors = 0

        for (i, flag) in enumerate(self.errflags):
            if errmask & flag:
                n_errors += 1
                self.errcounters[i] += ONE

        return n_errors

    def build_info(self):
        info = {name: count for (name, count) in zip(self.errnames, self.errcounters)}
        info["errors"] = self.errcounters.sum()
        info["is_success"] = self.result.get_is_victorious()
        return info
