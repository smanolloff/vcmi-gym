import gymnasium
from .envs.v0.vcmi_env import VcmiEnv, InfoDict
from .envs.v0.util.test_helper import TestHelper
from .envs.v0.util.vcmi_cnn import VcmiCNN
from .envs.v0.util.vcmi_cnn2 import VcmiCNN2
from .envs.v0.util.maskable_qrdqn.maskable_qrdqn import MaskableQRDQN

all = [VcmiEnv, VcmiCNN, VcmiCNN2, InfoDict, TestHelper, MaskableQRDQN]

gymnasium.register(id="VCMI-v0", entry_point="vcmi_gym:VcmiEnv")
