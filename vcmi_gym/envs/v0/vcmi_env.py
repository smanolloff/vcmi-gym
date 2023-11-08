import logging
import numpy as np
import gymnasium as gym

from .connector.pyconnector import PyConnector

# the numpy data type
# it seems pytorch is optimized for float32
DTYPE = np.float32


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        mapname,
        render_mode="ansi",
        vcmi_loglevel="error"
    ):
        self.render_mode = render_mode

        # NOTE: removing action=0 (retreat) for now
        #       => start from 1 and reduce total actions by 1
        # self.action_space = gym.spaces.Discrete(PyConnector.ACTION_MAX + 1)
        self.action_space = gym.spaces.Discrete(PyConnector.ACTION_MAX, start=1)
        self.observation_space = gym.spaces.Box(shape=(PyConnector.STATE_SIZE,), low=-1, high=1, dtype=DTYPE)
        self.pc = PyConnector(mapname, vcmi_loglevel)
        self.result = self.pc.start()

    def step(self, action):
        if self.result.is_battle_over():
            raise Exception("Reset needed")

        self.result = self.pc.act(action)
        reward = self.calc_reward(self.result)
        return self.result.state(), reward, self.result.is_battle_over(), False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.result, {}

    def render(self):
        if self.render_mode == "ansi":
            return self.pc.render_ansi()
        elif self.render_mode == "rgb_array":
            gym.logger.warn("Rendering RGB arrays not yet implemented for VcmiEnv")
        elif self.render_mode is None:
            gym.logger.warn("Cannot render with no render mode set")

        return

    def close(self):
        self.pc.shutdown()

    #
    # private
    #

    def calc_reward(self, res):
        n_errors = res.n_errors()
        logging.debug("Action errors: %d" % n_errors)

        if n_errors:
            return -10 * n_errors

        net_dmg = res.dmg_dealt() - res.dmg_received()
        net_value = res.value_killed() - res.value_lost()

        return net_value + 10*net_dmg
