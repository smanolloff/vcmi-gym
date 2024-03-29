# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

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
