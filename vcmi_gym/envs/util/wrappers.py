import gymnasium as gym


class LegacyActionSpaceWrapper(gym.Wrapper):
    """
    Simulates action offset for legacy envs where retreat
    action (0) was removed and instead action 1 became 0,
    2 became 1, etc
    """

    def action_mask(self):
        return self.env.result.actmask[1:]

    def step(self, action):
        return self.env.step(action + 1)
