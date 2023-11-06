import logging
import numpy as np
import gymnasium as gym

from .connector.pyconnector import PyConnector

# the numpy data type
# it seems pytorch is optimized for float32
DTYPE = np.float32


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["browser", "rgb_array"], "render_fps": 30}

    def __init__(self):
        self.render_mode = None
        self.action_space = gym.spaces.Discrete(1322)
        self.observation_space = gym.spaces.Box(shape=(334,), low=-1, high=1, dtype=DTYPE)
        self.pc = PyConnector()
        self.state = self.pc.start()

    def step(self, action):
        self.old_state = self.state
        self.state, action_was_valid, terminated = self.pc.act(action)
        reward = self.calc_reward(action_was_valid, self.state, self.old_state)
        return self.state, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.old_state:
            # do a reset only if at least 1 action was made
            self.old_state = None
            self.state = self.pc.reset()

        return self.state, {}

    def render(self, render_mode="browser"):
        gym.logger.warn("Rendering not yet implemented for VcmiEnv")
        return

    def close(self):
        self.pc.shutdown()

    #
    # private
    #

    def calc_reward(self, action_was_valid, new_state, old_state):
        # TODO
        return 0
