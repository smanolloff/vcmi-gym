from vcmi_gym.envs.v0.util.vcmi_nn.vcmi_ppo import VcmiPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack


# if __name__ == "__main__":
venv = make_vec_env("VCMI-v0", env_kwargs={"mapname": "ai/generated/A01.vmap"})
venv = VecFrameStack(venv, n_stack=2, channels_order="first")
venv.reset()
model = VcmiPPO(
    env=venv,
    batch_size=2,
    n_steps=4,
    policy="VcmiPolicy",
    policy_kwargs=dict(
        features_extractor_kwargs=dict(
            layers=[{"t": "Conv2d", "out_channels": 32, "kernel_size": (1, 15), "stride": (1, 15), "padding": 0}],
            activation="ReLU",
            output_dim=1024
        )
    )
)

model.learn(total_timesteps=1000)
