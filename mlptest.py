from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from vcmi_gym import VcmiCNN

if __name__ == '__main__':
    venv = make_vec_env("VCMI-v0", env_kwargs={"mapname": "ai/M1.vmap"})
    model = PPO(policy="MlpPolicy", env=venv)
    model.learn(total_timesteps=1000)
