import gymnasium
from .envs.v0.vcmi_env import VcmiEnv, InfoDict
from .envs.v0.util.test_helper import TestHelper
from .envs.v0.util.vcmi_nn import VcmiNN
from .envs.v0.util.maskable_qrdqn.maskable_qrdqn import MaskableQRDQN

all = [VcmiEnv, VcmiNN, InfoDict, TestHelper, MaskableQRDQN]

gymnasium.register(id="VCMI-v0", entry_point="vcmi_gym:VcmiEnv")
