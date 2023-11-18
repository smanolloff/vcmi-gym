import time
import numpy as np
import gymnasium as gym
import os

from .util import log
from .util.pyconnector import PyConnector, ERRNAMES

# the numpy data type (pytorch works best with float32)
DTYPE = np.float32
ZERO = DTYPE(0)
ONE = DTYPE(1)


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    VCMI_LOGLEVELS = ["trace", "debug", "info", "warn", "error"]
    INFO_KEYS = ERRNAMES + ["errors", "net_value", "is_success"]

    def __init__(
        self,
        mapname,
        seed=None,  # not used currently
        render_mode="ansi",
        render_each_step=False,
        vcmi_loglevel_global="error",
        vcmi_loglevel_ai="error",
        consecutive_error_reward_factor=-1
    ):
        assert vcmi_loglevel_global in VcmiEnv.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in VcmiEnv.VCMI_LOGLEVELS

        self.logger = log.get_logger("VcmiEnv", "DEBUG")

        self.connector = PyConnector(mapname, vcmi_loglevel_global, vcmi_loglevel_ai)
        (self.result, statesize, nactions, errflags) = self.connector.start()
        self.errflags = errflags

        # NOTE: removing action=0 (retreat) which is used for resetting...
        #       => start from 1 and reduce total actions by 1
        #          XXX: there seems to be a bug as start=1 causes this error:
        #          index 1322 is out of bounds for dimension 2 with size 1322
        #       => just start from 0, reduce max by 1, and manually add +1
        self.action_space = gym.spaces.Discrete(nactions - 1)
        self.observation_space = gym.spaces.Box(shape=(statesize,), low=-1, high=1, dtype=DTYPE)

        # <params>
        self.render_mode = render_mode
        self.render_each_step = render_each_step
        self.consecutive_error_reward_factor = consecutive_error_reward_factor
        # </params>

        self.n_steps = 0  # needed for reset
        self.reset(seed=seed)

    def step(self, action):
        if self.terminated:
            raise Exception("Reset needed")

        res = self.connector.act(action + 1)
        obs = res.state
        rew = self.calc_reward()
        term = res.is_battle_over
        info = self.build_info()

        self.result = res
        self.terminated = term
        self.n_steps += 1
        self.reward_last = rew
        self.reward_total += rew

        if self.render_each_step:
            if self.n_errors_last == 0:
                print("%s\nSkipped renders: %s" % (self.render(), self.n_renders_skipped))
                self.n_renders_skipped = 0
            else:
                self.n_renders_skipped += 1

        return obs, rew, term, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.n_steps > 0:
            self.result = self.connector.reset()

        self.n_steps = 0
        self.n_renders_skipped = 0
        self.n_errors_last = 0
        self.errcounters = np.zeros(len(ERRNAMES), dtype=DTYPE)
        self.value_total = 0
        self.reward_total = 0
        self.reward_last = 0
        self.terminated = False

        return self.result.state, self.build_info()

    def render(self):
        if self.render_mode == "ansi":
            footer = (
                f"Reward: {self.reward_last}\n"
                f"Total reward: {self.reward_total}\n"
                f"Total value: {self.value_total}\n"
                f"n_steps: {self.n_steps}"
            )

            return "%s\n%s" % (self.connector.render_ansi(), footer)

        elif self.render_mode == "rgb_array":
            gym.logger.warn("Rendering RGB arrays not yet implemented for VcmiEnv")
        elif self.render_mode is None:
            gym.logger.warn("Cannot render with no render mode set")

        return

    def close(self):
        # graceful shutdown of VCMI is not impossible (by design)
        # The 10+ running threads lead to segfaults on any such attempt anyway
        gym.logger.warn("Graceful shutdown is not supported by VCMI")
        # exit(1)

    def error_summary(self):
        res = "Error summary:\n"
        for i, name in enumerate(self.errnames):
            res += "%25s: %d\n" % (name, self.errcounters[i])

        return res

    #
    # private
    #

    def calc_reward(self):
        res = self.result
        self.n_errors_last = self.parse_errmask(res.errmask)
        # self.logger.debug("Action errors: %d" % n_errors)

        # Penalize with ever-increasing amounts to prevent
        # it from just making errors forever, avoiding any damage
        if self.n_errors_last > 0:
            self.n_errors_consecutive += self.n_errors_last
            return self.n_errors_consecutive * self.consecutive_error_reward_factor

        self.n_errors_consecutive = 0

        net_dmg = res.dmg_dealt - res.dmg_received
        net_value = res.value_killed - res.value_lost
        self.value_total += net_value

        # XXX: pikemen value=80, life=10
        return net_value + 5 * net_dmg

    def parse_errmask(self, errmask):
        n_errors = 0

        for i, flag in enumerate(self.errflags):
            if errmask & flag:
                n_errors += 1
                self.errcounters[i] += ONE

        return n_errors

    def build_info(self):
        info = {name: count for (name, count) in zip(ERRNAMES, self.errcounters)}
        info["errors"] = self.errcounters.sum()
        info["net_value"] = self.value_total
        info["is_success"] = self.result.is_victorious
        return info
