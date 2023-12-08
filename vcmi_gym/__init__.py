import gymnasium
from .envs.v0.vcmi_env import VcmiEnv, InfoDict
from .envs.v0.util.test_helper import TestHelper
from .envs.v0.util.vcmi_cnn import VcmiCNN

all = [VcmiEnv, VcmiCNN, InfoDict, TestHelper]

gymnasium.register(id="VCMI-v0", entry_point="vcmi_gym:VcmiEnv")
