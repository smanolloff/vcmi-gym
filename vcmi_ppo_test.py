from vcmi_gym.envs.v0.util.vcmi_nn.vcmi_ppo import VcmiPPO
from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    venv = make_vec_env("VCMI-v0", env_kwargs={"mapname": "ai/generated/A01.vmap"})
    venv.reset()
    model = VcmiPPO(
        env=venv,
        batch_size=4,
        n_steps=8,
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
