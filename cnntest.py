from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from vcmi_gym import VcmiCNN

import gymnasium as gym


class AtariMimic(gym.Wrapper):
    # Mimic grayscale 84x84 image

    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=env.observation_space.dtype
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation_space.sample
        return obs, reward, terminated, truncated, info


if __name__ == '__main__':
    venv = make_vec_env("VCMI-v0", env_kwargs={"mapname": "ai/M1.vmap"})
    model = PPO(policy="CnnPolicy", env=venv, policy_kwargs=dict(features_extractor_class=VcmiCNN))
    model.learn(total_timesteps=1000)
