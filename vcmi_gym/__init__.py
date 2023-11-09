import gymnasium
from .envs.v0.vcmi_env import VcmiEnv

all = [VcmiEnv]

gymnasium.register(id="VCMI-v0", entry_point="vcmi_gym:VcmiEnv")
