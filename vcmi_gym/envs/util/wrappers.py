import gymnasium as gym


class LegacyActionSpaceWrapper(gym.Wrapper):
    """
    Simulates action offset for legacy envs where retreat
    action (0) was removed and instead action 1 became 0,
    2 became 1, etc
    """

    @property
    def action_space(self):
        return gym.spaces.Discrete(n=self.env.action_space.n - 1)

    def action_mask(self):
        return self.env.action_mask()[1:]

    def step(self, action):
        return self.env.step(action + 1)


class LegacyObservationSpaceWrapper(gym.Wrapper):
    """
    Converts dict-observation space (with keys "observation", "action_mask")
    to plain observation space (value from "observation"), as well as
    adds the `action_mask()` method.
    """

    @property
    def observation_space(self):
        return self.env.observation_space["observation"]

    def action_mask(self):
        return self._dict_obs["action_mask"]

    def links(self):
        return self._dict_obs["links"]

    def step(self, *args, **kwargs):
        obs, *rest = self.env.step(*args, **kwargs)
        self._dict_obs = obs
        return obs["observation"], *rest

    def reset(self, *args, **kwargs):
        obs, *rest = self.env.reset(*args, **kwargs)
        self._dict_obs = obs
        return obs["observation"], *rest
