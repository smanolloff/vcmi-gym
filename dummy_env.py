import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env


class DummyEnv(gym.Env):
    def __init__(self, name="?", n=10, minterm=None, maxterm=None):
        self.name = name
        self.action_space = gym.spaces.Discrete(n)
        self.observation_space = gym.spaces.Box(shape=(1,), low=0, high=n-1, dtype=int)
        self.minterm = minterm or n
        self.maxterm = maxterm or n
        self._step = 0

    def _log(self, msg):
        print("*** DUMMY ENV %s: %s" % (self.name, msg))

    def step(self, action):
        if self._terminated:
            raise Exception("reset needed")

        self._step += 1
        obs = np.array([action])
        rew = action
        self._terminated = self._step >= self.minterm and np.random.randint(self.maxterm - self._step + 1) == 0
        self._log("[%d] step(action=%s) term=%s" % (self._step, action, self._terminated))
        return obs, rew, self._terminated, False, {}

    def reset(self, *args, **kwargs):
        self._log("reset()")
        self._step = 0
        self._terminated = 0
        return np.array([0]), {}


def create_venv(n_envs, env_kwargs={}):
    state = {"n": 0}

    def env_creator(**_env_kwargs):
        assert state["n"] < n_envs
        env_kwargs2 = dict(env_kwargs, name="env.%d" % state["n"])
        state["n"] += 1
        return DummyEnv(**env_kwargs2)

    return make_vec_env(env_creator, n_envs=n_envs)


gym.register(id="Dummy-v0", entry_point="dummy_env:DummyEnv")

# from sb3_contrib import RecurrentPPO
# import dummy_env; import gymnasium as gym; venv = dummy_env.create_venv(n_envs=2, env_kwargs=dict(minterm=5, maxterm=10))
# model = RecurrentPPO(env=venv, policy="MlpLstmPolicy")
