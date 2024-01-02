import gymnasium
from .envs.v0.vcmi_env import VcmiEnv, InfoDict
from .envs.v0.util.test_helper import TestHelper
from .envs.v0.util.maskable_qrdqn.maskable_qrdqn import MaskableQRDQN
from .envs.v0.util.vcmi_nn import (
    VcmiFeaturesExtractor,
    VcmiPolicy,
    VcmiPPO,
)

all = [
    VcmiEnv,
    InfoDict,
    TestHelper,
    MaskableQRDQN,
    VcmiFeaturesExtractor,
    VcmiPolicy,
    VcmiPPO,
]

gymnasium.register(id="VCMI-v0", entry_point="vcmi_gym:VcmiEnv")
