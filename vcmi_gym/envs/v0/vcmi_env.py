import logging
import numpy as np
import gymnasium as gym

from .connector.build import connector

# the numpy data type (pytorch works best with float32)
DTYPE = np.float32
ERROR_MAPPING = connector.get_error_mapping()


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        mapname,
        render_mode="ansi",
        vcmi_loglevel="error"
    ):
        self.render_mode = render_mode
        self.errcounters = {key: 0 for key in ERROR_MAPPING.keys()}

        # NOTE: removing action=0 (retreat) which is used for resetting...
        #       => start from 1 and reduce total actions by 1
        self.action_space = gym.spaces.Discrete(connector.get_n_actions() - 1, start=1)
        self.observation_space = gym.spaces.Box(shape=(connector.get_state_size(),), low=-1, high=1, dtype=DTYPE)
        self.connector = connector.Connector(mapname, vcmi_loglevel)
        self.result = self.connector.start()

    def step(self, action):
        if self.result.get_is_battle_over():
            raise Exception("Reset needed")

        self.result = self.connector.act(action)
        reward = self.calc_reward(self.result)
        return self.result.get_state(), reward, self.result.get_is_battle_over(), False, {}
        return np.zeros(334, dtype=DTYPE), 0, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.result = self.connector.reset()
        return self.result.get_state(), {}

    def render(self):
        if self.render_mode == "ansi":
            return self.connector.render_ansi()
        elif self.render_mode == "rgb_array":
            gym.logger.warn("Rendering RGB arrays not yet implemented for VcmiEnv")
        elif self.render_mode is None:
            gym.logger.warn("Cannot render with no render mode set")

        return

    def close(self):
        # self.connector.shutdown()
        pass

    def error_summary(self):
        res = "Error summary:\n"
        for (flag, count) in self.errcounters.items():
            err, _msg = ERROR_MAPPING[flag]
            res += ("%25s: %d\n" % (err, count))

        return res

    #
    # private
    #

    def calc_reward(self, res):
        n_errors = self.parse_errmask(res.get_errmask())
        logging.debug("Action errors: %d" % n_errors)

        if n_errors:
            return -10 * n_errors

        net_dmg = res.get_dmg_dealt() - res.get_dmg_received()
        net_value = res.get_value_killed() - res.get_value_lost()

        return net_value + 10*net_dmg

    def parse_errmask(self, errmask):
        n_errors = 0

        for flag in self.errcounters.keys():
            if errmask & flag:
                n_errors += 1
                self.errcounters[flag] += 1

        return n_errors
