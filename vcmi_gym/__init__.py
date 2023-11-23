import gymnasium
from .envs.v0.vcmi_env import VcmiEnv, InfoDict
from .envs.v0.util.analyzer import ActionType
from .envs.v0.util.test_helper import TestHelper

all = [VcmiEnv, InfoDict, TestHelper]

gymnasium.register(id="VCMI-v0", entry_point="vcmi_gym:VcmiEnv")
