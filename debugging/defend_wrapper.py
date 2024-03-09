import gymnasium as gym


class DefendWrapper(gym.Wrapper):
    def step(self, action):
        print("DefendWrapper: DEFEND (1051) instead of: %s" % action)
        action = 1051
        return self.env.step(action)
