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
# distanceributed under the License is distanceributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import gymnasium
from .envs.v1.vcmi_env import VcmiEnv as VcmiEnv_v1, InfoDict
from .envs.v1.decoder.other import (
    HexAction,
    HexState,
    MeleeDistance,
    ShootDistance,
    DmgMod,
    Side,
)

from .envs.v2.vcmi_env import VcmiEnv as VcmiEnv_v2

from .envs.util.dual_env import DualEnvController, DualEnvClient
from .envs.util.wrappers import LegacyActionSpaceWrapper
from .tools.test_helper import TestHelper

all = [
    VcmiEnv_v1,
    VcmiEnv_v2,
    DualEnvController,
    DualEnvClient,
    HexAction,
    HexState,
    MeleeDistance,
    ShootDistance,
    DmgMod,
    Side,
    InfoDict,
    TestHelper,
    LegacyActionSpaceWrapper,
]

gymnasium.register(id="VCMI-v1", entry_point="vcmi_gym:VcmiEnv")
