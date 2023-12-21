from vcmi_gym.envs.v0.util.maskable_qrdqn.maskable_qrdqn import MaskableQRDQN
from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    venv = make_vec_env("VCMI-v0", env_kwargs={"mapname":"ai/generated/A01.vmap"})
    venv.reset()
    model = MaskableQRDQN(env=venv, policy="MlpPolicy", buffer_size=1000)
    model.learn(total_timesteps=1000)
