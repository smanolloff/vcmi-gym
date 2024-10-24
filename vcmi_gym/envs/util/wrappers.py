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


class LegacyObservationSpaceWrapper(gym.Wrapper):
    """
    Converts dict-observation space (with keys "observation", "action_mask")
    to plain observation space (value from "observation"), as well as
    adds an `action_mask()` method.
    """

    @property
    def observation_space(self):
        return self.env.observation_space["observation"]

    def action_mask(self):
        return self.env.result.actmask

    def step(self, *args, **kwargs):
        obs, *rest = self.env.step(*args, **kwargs)
        return obs["observation"], *rest

    def reset(self, *args, **kwargs):
        obs, *rest = self.env.reset(*args, **kwargs)
        return obs["observation"], *rest
