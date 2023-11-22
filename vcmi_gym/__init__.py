import gymnasium
from .envs.v0.vcmi_env import VcmiEnv
from .envs.v0.util.test_helper import TestHelper

all = [VcmiEnv, TestHelper]

gymnasium.register(id="VCMI-v0", entry_point="vcmi_gym:VcmiEnv")
