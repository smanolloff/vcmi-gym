import logging
import numpy as np
import gymnasium as gym

from .connector.pyconnector import PyConnector

# the numpy data type
# it seems pytorch is optimized for float32
DTYPE = np.float32


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["browser", "rgb_array"], "render_fps": 30}

    def __init__(self, mapname):
        self.render_mode = None

        # NOTE: removing action=0 (retreat) for now
        #       => start from 1 and reduce total actions by 1
        # self.action_space = gym.spaces.Discrete(PyConnector.ACTION_MAX + 1)
        self.action_space = gym.spaces.Discrete(PyConnector.ACTION_MAX, start=1)
        self.observation_space = gym.spaces.Box(shape=(PyConnector.STATE_SIZE,), low=-1, high=1, dtype=DTYPE)
        self.pc = PyConnector(mapname)
        self.result = self.pc.start()

    def step(self, action):
        if self.result.is_battle_over():
            raise Exception("Reset needed")

        self.old_result = self.result
        self.result = self.pc.act(action)
        reward = self.calc_reward(self.result, self.old_result)
        return self.result.state(), reward, self.result.is_battle_over(), False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.old_result:
            # do a reset only if at least 1 action was made
            self.old_result = None
            self.result = self.pc.reset()

        return self.result, {}

    def render(self, render_mode="browser"):
        gym.logger.warn("Rendering not yet implemented for VcmiEnv")
        return

    def close(self):
        self.pc.shutdown()

    #
    # private
    #

    def calc_reward(self, result, old_result):
        n_errors = result.n_errors()
        logging.debug("Last action had %d errors" % n_errors)

        reward = -n_errors + result.dmg_dealt() - result.dmg_received()
        return reward
